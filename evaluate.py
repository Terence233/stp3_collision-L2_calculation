from argparse import ArgumentParser
from PIL import Image
import torch
import torch.utils.data
import numpy as np
import torchvision
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import matplotlib
from matplotlib import pyplot as plt
import pathlib
import datetime

from stp3.datas.NuscenesData import FuturePredictionDataset
from stp3.trainer import TrainingModule
from stp3.metrics import IntersectionOverUnion, PanopticMetric, PlanningMetric
from stp3.utils.network import preprocess_batch, NormalizeInverse
from stp3.utils.instance import predict_instance_segmentation_and_trajectories
from stp3.utils.visualisation import make_contour

###################################################
# 为了移植resnet/你的模型需要额外加载的库
# from networks.resnet_nuscenes import ResNetModel_C
# import torchvision.transforms as tfs
##################################################

def mk_save_dir():
    now = datetime.datetime.now()
    string = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
    save_path = pathlib.Path('imgs') / string
    save_path.mkdir(parents=True, exist_ok=False)
    return save_path

def eval(checkpoint_path, dataroot):
    save_path = mk_save_dir()

    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model
    
    ##########################################################
    model2 = torch.load('./1.pt').to(device) # 选择你要加载的模型权重文件
    model2 = model2.eval()
    ##########################################################

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1
    cfg.LIFT.GT_DEPTH = False
    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.MAP_FOLDER = dataroot

    dataroot = cfg.DATASET.DATAROOT
    nworkers = cfg.N_WORKERS
    ##################################################################################################
    # nusc = NuScenes(version='v1.0-{}'.format(cfg.DATASET.VERSION), dataroot=dataroot, verbose=False)
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=False)  # 选择你的数据集版本: v1.0-mini, v1.0-trainval, v1.0-test
    ##################################################################################################
    valdata = FuturePredictionDataset(nusc, 1, cfg)
    valloader = torch.utils.data.DataLoader(
        valdata, batch_size=cfg.BATCHSIZE, shuffle=False, num_workers=nworkers, pin_memory=True, drop_last=False
    )

    n_classes = len(cfg.SEMANTIC_SEG.VEHICLE.WEIGHTS)
    hdmap_class = cfg.SEMANTIC_SEG.HDMAP.ELEMENTS
    metric_vehicle_val = IntersectionOverUnion(n_classes).to(device)
    future_second = int(cfg.N_FUTURE_FRAMES / 2)

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        metric_pedestrian_val = IntersectionOverUnion(n_classes).to(device)

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        metric_hdmap_val = []
        for i in range(len(hdmap_class)):
            metric_hdmap_val.append(IntersectionOverUnion(2, absent_score=1).to(device))

    if cfg.INSTANCE_SEG.ENABLED:
        metric_panoptic_val = PanopticMetric(n_classes=n_classes).to(device)

    if cfg.PLANNING.ENABLED:
        metric_planning_val = []
        for i in range(future_second):
            metric_planning_val.append(PlanningMetric(cfg, 2*(i+1)).to(device))


    for index, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image'] # 形状是: [batch_size, sequence length, n_cameras, C, H, W], 具体说明见NuscenesData.py row 575   例子：[1,3,6,3,224,480]
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']   # shape: [batch_size, sequence length, 6] 例子：[1, 9, 6].   但加载数据之后，stp3模型只会用到B,T,3(只用x,y,z，不用旋转的量)
        # 第二个维度是9是因为过去有3帧，未来有6帧
        # 第三个维度是6是因为6-dof pose vector comprising translation components (tx, ty, tz) and rotation components (rx, ry, rz)注意这里不是四元数而是欧拉角
        command = batch['command']
        trajs = batch['sample_trajectory']
        target_points = batch['target_point']
        B = len(image)
        labels = trainer.prepare_future_labels(batch)

        with torch.no_grad():
            output = model(
                image, intrinsics, extrinsics, future_egomotion
            )
        ##################################################################################
        # 在这里修改待加载的数据，以满足你的模型的输入要求
        images_raw = batch['images_2']   # 形状是: [batch_size, sequence length, n_cameras, C, H, W] 例子: [1,3,6,3,384,704]. 是经过你想要的预处理后的图片
        image2 = torch.squeeze(images_raw[:, :, 0:1, :, :, :].contiguous(), dim=2) # 第三个维度的切片是选择哪个方向的相机(现在选的是cam_front，但真的是不是还没有验证), 然后把这个维度压缩掉
        # 例子: [1,3,3,900,1600]

        # 如果要用你自己的预处理方式，请在NuscenesData.py中的FuturePredictionDataset类中的get_input_data函数中修改
        # 对于周博的模型：
        # target image shape for the model: [batch_size, sequence length, C, H, W]， 例子：[1,3,3,384,704]
        # n_cameras is ignored because of the model doesnt support more than 1 camera
        
        # 在这里修改ego vehicle的未来运动信息，以满足你的模型的输入要求
        # 对于周博的模型，只接受egomotion shape as (batch_size, n_future_frames*2),
        f233 = future_egomotion[:, :3, :2].squeeze(0).contiguous()  # shape: (input_n, 2) 例子：[3(前n帧), 2（x,y）]
        future_egomotion2 = torch.tensor([[f233[1,0]-f233[0,0],f233[1,1]-f233[0,1],f233[2,0]-f233[1,0],f233[2,1]-f233[1,1]]], dtype=torch.float32).to(device)  # shape: (B, 4) 1, 4, 1

        # 将处理后的输入数据传入模型
        output2 = model2([image2.float(), future_egomotion2.float()]) # 这里是第二个模型的输出, 对于周博的模型，输出形状是[2xpredict_n, 1]

        # 在这里修改以处理模型的输出，使其满足stp3模型的轨迹输出要求，即[batch_size, n_future_frames, 3]
        # 创建一个形状为6x1的0值tensor
        zeros_tensor = torch.zeros(6, 1).to(device)
        # 沿着最后一个维度拼接  
        final_traj_ours = torch.cat((output2.reshape([6,2]), zeros_tensor), dim=-1).to(device).unsqueeze(0)  # shape should be torch.Size([1, 6(未来帧数), 3(x,y,z)])
        ###################################################################################

        n_present = model.receptive_field

        # semantic segmentation metric
        seg_prediction = output['segmentation'].detach()
        seg_prediction = torch.argmax(seg_prediction, dim=2, keepdim=True)
        metric_vehicle_val(seg_prediction[:, n_present - 1:], labels['segmentation'][:, n_present - 1:])

        if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
            pedestrian_prediction = output['pedestrian'].detach()
            pedestrian_prediction = torch.argmax(pedestrian_prediction, dim=2, keepdim=True)
            metric_pedestrian_val(pedestrian_prediction[:, n_present - 1:],
                                       labels['pedestrian'][:, n_present - 1:])
        else:
            pedestrian_prediction = torch.zeros_like(seg_prediction)

        if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
            for i in range(len(hdmap_class)):
                hdmap_prediction = output['hdmap'][:, 2 * i:2 * (i + 1)].detach()
                hdmap_prediction = torch.argmax(hdmap_prediction, dim=1, keepdim=True)
                metric_hdmap_val[i](hdmap_prediction, labels['hdmap'][:, i:i + 1])

        if cfg.INSTANCE_SEG.ENABLED:
            pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
                output, compute_matched_centers=False, make_consistent=True
            )
            metric_panoptic_val(pred_consistent_instance_seg[:, n_present - 1:],
                                     labels['instance'][:, n_present - 1:])

        if cfg.PLANNING.ENABLED:
            occupancy = torch.logical_or(seg_prediction, pedestrian_prediction)
            _, final_traj_stp3 = model.planning(
                cam_front=output['cam_front'].detach(),
                trajs=trajs[:, :, 1:],
                gt_trajs=labels['gt_trajectory'][:, 1:],
                cost_volume=output['costvolume'][:, n_present:].detach(),
                semantic_pred=occupancy[:, n_present:].squeeze(2),
                hd_map=output['hdmap'].detach(),
                commands=command,
                target_points=target_points
            )

            occupancy = torch.logical_or(labels['segmentation'][:, n_present:].squeeze(2),
                                         labels['pedestrian'][:, n_present:].squeeze(2))
            for i in range(future_second):
                cur_time = (i+1)*2
                ###############################################################################################################################
                # 在这里切换输出stp3的碰撞率还是你的模型的碰撞率
                #metric_planning_val[i](final_traj_stp3[:,:cur_time].detach(), labels['gt_trajectory'][:,1:cur_time+1], occupancy[:,:cur_time])
                metric_planning_val[i](final_traj_ours[:,:cur_time].detach(), labels['gt_trajectory'][:,1:cur_time+1], occupancy[:,:cur_time])
                ###############################################################################################################################
        if index % 100 == 0:
            save(output, labels, batch, n_present, index, save_path)


    results = {}

    scores = metric_vehicle_val.compute()
    results['vehicle_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.PEDESTRIAN.ENABLED:
        scores = metric_pedestrian_val.compute()
        results['pedestrian_iou'] = scores[1]

    if cfg.SEMANTIC_SEG.HDMAP.ENABLED:
        for i, name in enumerate(hdmap_class):
            scores = metric_hdmap_val[i].compute()
            results[name + '_iou'] = scores[1]

    if cfg.INSTANCE_SEG.ENABLED:
        scores = metric_panoptic_val.compute()
        for key, value in scores.items():
            results['vehicle_'+key] = value[1]

    if cfg.PLANNING.ENABLED:
        for i in range(future_second):
            scores = metric_planning_val[i].compute()
            for key, value in scores.items():
                results['plan_'+key+'_{}s'.format(i+1)]=value.mean()

    for key, value in results.items():
        print(f'{key} : {value.item()}')     # 在这里打印每个benchmark的结果

def save(output, labels, batch, n_present, frame, save_path):
    hdmap = output['hdmap'].detach()
    segmentation = output['segmentation'][:, n_present - 1].detach()
    pedestrian = output['pedestrian'][:, n_present - 1].detach()
    gt_trajs = labels['gt_trajectory']
    images = batch['image']

    denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
    )

    val_w = 2.99
    val_h = 2.99 * (224. / 480.)
    plt.figure(1, figsize=(4*val_w,2*val_h))
    width_ratios = (val_w,val_w,val_w,val_w)
    gs = matplotlib.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    plt.subplot(gs[0, 0])
    plt.annotate('FRONT LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,0].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 1])
    plt.annotate('FRONT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,1].cpu()))
    plt.axis('off')

    plt.subplot(gs[0, 2])
    plt.annotate('FRONT RIGHT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    plt.imshow(denormalise_img(images[0,n_present-1,2].cpu()))
    plt.axis('off')

    plt.subplot(gs[1, 0])
    plt.annotate('BACK LEFT', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0,n_present-1,3].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[1, 1])
    plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 4].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[1, 2])
    plt.annotate('BACK', (0.01, 0.87), c='white', xycoords='axes fraction', fontsize=14)
    showing = denormalise_img(images[0, n_present - 1, 5].cpu())
    showing = showing.transpose(Image.FLIP_LEFT_RIGHT)
    plt.imshow(showing)
    plt.axis('off')

    plt.subplot(gs[:, 3])
    showing = torch.zeros((200, 200, 3)).numpy()
    showing[:, :] = np.array([219 / 255, 215 / 255, 215 / 255])

    # drivable
    area = torch.argmax(hdmap[0, 2:4], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([161 / 255, 158 / 255, 158 / 255])

    # lane
    area = torch.argmax(hdmap[0, 0:2], dim=0).cpu().numpy()
    hdmap_index = area > 0
    showing[hdmap_index] = np.array([84 / 255, 70 / 255, 70 / 255])

    # semantic
    semantic_seg = torch.argmax(segmentation[0], dim=0).cpu().numpy()
    semantic_index = semantic_seg > 0
    showing[semantic_index] = np.array([255 / 255, 128 / 255, 0 / 255])

    pedestrian_seg = torch.argmax(pedestrian[0], dim=0).cpu().numpy()
    pedestrian_index = pedestrian_seg > 0
    showing[pedestrian_index] = np.array([28 / 255, 81 / 255, 227 / 255])

    plt.imshow(make_contour(showing))
    plt.axis('off')

    bx = np.array([-50.0 + 0.5/2.0, -50.0 + 0.5/2.0])
    dx = np.array([0.5, 0.5])
    w, h = 1.85, 4.084
    pts = np.array([
        [-h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, w / 2.],
        [h / 2. + 0.5, -w / 2.],
        [-h / 2. + 0.5, -w / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

    plt.xlim((200, 0))
    plt.ylim((0, 200))
    gt_trajs[0, :, :1] = gt_trajs[0, :, :1] * -1
    gt_trajs = (gt_trajs[0, :, :2].cpu().numpy() - bx) / dx
    plt.plot(gt_trajs[:, 0], gt_trajs[:, 1], linewidth=3.0)

    plt.savefig(save_path / ('%04d.png' % frame))
    plt.close()

if __name__ == '__main__':
    parser = ArgumentParser(description='STP3 evaluation')
    parser.add_argument('--checkpoint', default='last.ckpt', type=str, help='path to checkpoint')
    # parser.add_argument('--dataroot', default=None, type=str)
    parser.add_argument('--dataroot', default='/data/sets/nuscenes', type=str)

    args = parser.parse_args()

    eval(args.checkpoint, args.dataroot)
