#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# 导入必要的库和模块
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

# 尝试导入TensorBoard，如果导入失败则设置TENSORBOARD_FOUND为False
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# 尝试导入fused_ssim，如果导入失败则设置FUSED_SSIM_AVAILABLE为False
try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

# 尝试导入SparseGaussianAdam，如果导入失败则设置SPARSE_ADAM_AVAILABLE为False
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

# 定义训练函数
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """
    Trains a Gaussian Splatting model using the provided dataset and options.
    Args:
        dataset: The dataset object containing training data and configurations.
        opt: An object containing various training options and hyperparameters.
        pipe: The rendering pipeline object.
        testing_iterations: List of iterations at which testing should be performed.
        saving_iterations: List of iterations at which the model should be saved.
        checkpoint_iterations: List of iterations at which checkpoints should be saved.
        checkpoint: Path to a checkpoint file to resume training from.
        debug_from: Iteration number from which debugging should start.
    Returns:
        None
    """

    # 如果选择了稀疏Adam优化器但未安装相应的库，则退出程序
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    # 初始化一些变量
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)  # 准备输出和日志记录器
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)  # 创建高斯模型
    scene = Scene(dataset, gaussians)  # 创建场景
    gaussians.training_setup(opt)  # 设置高斯模型的训练参数
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)  # 加载检查点
        gaussians.restore(model_params, opt)  # 恢复模型参数

    # 设置背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建CUDA事件用于计时
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # 检查是否使用稀疏Adam优化器
    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)  # 获取深度L1权重的指数学习率函数

    # 获取训练相机的视点堆栈和索引
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        # 当网络GUI连接存在时，持续执行循环
        while network_gui.conn != None:
            try:
                # 初始化网络图像字节为None
                net_image_bytes = None
                
                # 从网络GUI接收多个参数：
                # - custom_cam: 自定义相机视角
                # - do_training: 是否继续训练
                # - pipe.convert_SHs_python: 是否使用Python转换球谐函数
                # - pipe.compute_cov3D_python: 是否使用Python计算3D协方差
                # - keep_alive: 保持连接活跃
                # - scaling_modifer #缩放修改器
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                
                # 如果收到自定义相机参数
                if custom_cam != None:
                    # 使用当前参数渲染图像
                    net_image = render(custom_cam, gaussians, pipe, background, 
                                        scaling_modifier=scaling_modifer, 
                                        use_trained_exp=dataset.train_test_exp, 
                                        separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    
                    # 将渲染图像转换为字节格式：
                    # 1. 将像素值限制在[0,1]范围内
                    # 2. 乘以255转换为8位格式
                    # 3. 转换为字节类型
                    # 4. 调整维度顺序（permute）
                    # 5. 确保内存连续（contiguous）
                    # 6. 转换为numpy数组
                    # 7. 转换为内存视图（memoryview）
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    
                # 发送渲染的图像数据和数据源路径到网络GUI
                network_gui.send(net_image_bytes, dataset.source_path)
                
                # 如果需要训练且（迭代次数小于最大迭代次数或不需要保持活跃），则跳出循环
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
                    
            # 如果发生异常，断开网络GUI连接
            except Exception as e:
                network_gui.conn = None
        # end of while loop

        iter_start.record()

        # 负责在训练过程中动态调整学习率
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加一次SH的级别，直到达到最大程度
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个相机视点
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # 渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # 将原始图像数据转移到CUDA设备(GPU)上
        gt_image = viewpoint_cam.original_image.cuda()

        # 计算预测图像和真实图像之间的L1损失
        # L1损失是绝对值差异的平均值，用于衡量两个图像的像素级差异
        Ll1 = l1_loss(image, gt_image)

        # 检查是否可以使用融合版本的SSIM(结构相似性)计算
        if FUSED_SSIM_AVAILABLE:
            # 使用融合版本的SSIM计算结构相似性
            # unsqueeze(0)在第0维添加一个维度，将图像转换为批处理格式
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            # 如果不可用融合版本，则使用标准SSIM计算
            # SSIM用于评估两个图像的结构相似度，范围在[-1,1]之间，1表示完全相同
            ssim_value = ssim(image, gt_image)

        # L1损失部分: (1.0 - opt.lambda_dssim) * Ll1
        # Ll1表示像素级别的绝对差异
        # (1.0 - opt.lambda_dssim)是L1损失的权重
        # SSIM损失部分: opt.lambda_dssim * (1.0 - ssim_value)
        # (1.0 - ssim_value)将SSIM转换为损失（因为SSIM值越大越好）
        # opt.lambda_dssim是SSIM损失的权重
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # 初始化深度L1损失的原始值为0
        Ll1depth_pure = 0.0

        # 判断是否需要计算深度损失:
        # 1. depth_l1_weight(iteration) > 0: 当前迭代的深度权重大于0
        # 2. viewpoint_cam.depth_reliable: 当前视角相机的深度数据可靠
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            # 从渲染包中获取预测的逆深度图
            invDepth = render_pkg["depth"]
            # 获取相机的真实逆深度图并移至GPU
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            # 获取深度遮罩并移至GPU
            depth_mask = viewpoint_cam.depth_mask.cuda()

            # 计算纯深度L1损失:
            # 1. (invDepth - mono_invdepth): 预测深度与真实深度的差异
            # 2. * depth_mask: 仅考虑有效深度区域
            # 3. torch.abs(): 取绝对值
            # 4. mean(): 计算平均值
            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            
            # 将纯深度损失乘以权重系数
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            
            # 将深度损失加入总损失
            loss += Ll1depth
            
            # 将深度损失转换为Python标量
            Ll1depth = Ll1depth.item()
        else:
            # 如果不计算深度损失，设置为0
            Ll1depth = 0

        # backward()将计算loss对所有requires_grad=True的张量的梯度
        # 具体步骤
        # 1. 构建计算图
        # PyTorch在前向传播时自动构建计算图
        # 记录所有计算操作和中间结果
        # 2. 梯度计算
        # 使用链式法则计算每个参数的梯度
        # 从损失函数开始，向后传播梯度
        # 3. 梯度累积
        # 对每个requires_grad=True的张量
        # 将计算得到的梯度存储在.grad属性中

        # 在gaussian splatting中，这行代码计算：
        # 1. 渲染损失的梯度
        # 2. 深度损失的梯度（如果启用）
        # 3. SSIM损失的梯度 这些梯度将用于更新3D高斯体的参数
        loss.backward()

        iter_end.record()

        # 使用torch.no_grad()上下文管理器，暂时关闭梯度计算，节省内存
        with torch.no_grad():
            # 使用指数移动平均(EMA)更新损失值，用于日志记录
            # 0.4是新值的权重，0.6是历史值的权重
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            # 每10次迭代更新进度条
            if iteration % 10 == 0:
                # 设置进度条后缀，显示当前损失值
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}", 
                    "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"
                })
                progress_bar.update(10)
            
            # 迭代结束时关闭进度条
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录训练报告并保存模型
            training_report(
                tb_writer,          # TensorBoard写入器
                iteration,          # 当前迭代次数
                Ll1,               # L1损失
                loss,              # 总损失
                l1_loss,           # L1损失函数
                iter_start.elapsed_time(iter_end),  # 迭代耗时
                testing_iterations,  # 测试迭代点
                scene,             # 场景对象
                render,            # 渲染函数
                (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),  # 渲染参数
                dataset.train_test_exp  # 训练测试经验
            )

            # 在指定迭代点保存高斯模型
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 密集化处理
            if iteration < opt.densify_until_iter:
                # 更新可见高斯体的最大2D半径
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], 
                    radii[visibility_filter]
                )
                # 添加密集化统计信息
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 在指定迭代区间执行密集化和剪枝
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 如果迭代次数超过不透明度重置间隔，则设置大小阈值为20
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 执行密集化和剪枝操作
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,  # 密集化梯度阈值
                        0.005,                       # 最小密度阈值
                        scene.cameras_extent,        # 相机范围
                        size_threshold,              # 大小阈值
                        radii                        # 半径信息
                    )
                
                # 在指定间隔或特定条件下重置不透明度
                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    gaussians.reset_opacity()

            # 更新优化器
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            # 保存检查点
            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

# 准备输出和日志记录器
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 设置输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建TensorBoard记录器
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# 训练报告
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 报告测试和训练集的样本
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
