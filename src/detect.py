# -*- coding: utf-8 -*-
"""
YOLO 摄像头实时目标检测

功能：
  1. 打开摄像头，实时读帧
  2. 用 YOLO 预训练模型检测画面中的物体
  3. 画检测框、类别标签、置信度
  4. 显示 FPS
  5. 按 q 或 ESC 退出，按 s 保存截图

用法：
    python detect.py                    # 默认摄像头（索引 0）+ yolo11n
    python detect.py --source 1         # 第二个摄像头
    python detect.py --model yolo11s.pt # 换更大的模型（更准但更慢）
    python detect.py --conf 0.5         # 提高置信度阈值（只显示更确定的）
    python detect.py --no-display       # 不开窗口（服务器/测试用），跑 N 帧后退出
    python detect.py --save out.jpg     # 跑一帧后保存并退出（截图模式）
"""
import argparse
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def parse_args():
    p = argparse.ArgumentParser(description="YOLO 摄像头实时检测")
    p.add_argument("--source", default="0",
                   help="摄像头索引（0/1/...）或视频文件路径")
    p.add_argument("--model", default="yolo11n.pt",
                   help="模型权重（yolo11n/s/m/l/x.pt，首次运行会自动下载）")
    p.add_argument("--conf", type=float, default=0.25,
                   help="置信度阈值（默认 0.25）")
    p.add_argument("--imgsz", type=int, default=640,
                   help="推理分辨率（默认 640，越小越快）")
    p.add_argument("--device", default=None,
                   help="推理设备：'0' = 第一块 GPU，'cpu' = 强制 CPU，"
                        "不填 = 自动（有 GPU 用 GPU）")
    p.add_argument("--no-display", action="store_true",
                   help="不开窗口（无 GUI 环境用）")
    p.add_argument("--max-frames", type=int, default=0,
                   help="跑多少帧后退出（0 = 一直跑）")
    p.add_argument("--save", default=None,
                   help="保存一张带标注的图到这个路径后退出")
    return p.parse_args()


def main():
    args = parse_args()

    # 延迟导入：这样 --help 不需要装好依赖
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请先安装: pip install ultralytics opencv-python")
        return 1

    print("=" * 62)
    print("  YOLO 摄像头实时检测")
    print("=" * 62)
    print(f"  模型:     {args.model}")
    print(f"  输入源:   {args.source}")
    print(f"  置信度:   {args.conf}")
    print(f"  分辨率:   {args.imgsz}")
    print()

    # ---- 1. 加载模型 ----
    print("[1/3] 加载模型（首次运行会自动下载权重）...")
    t0 = time.time()
    model = YOLO(args.model)

    # 设备选择：显式指定 > 自动（有 GPU 就用）
    import torch
    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = "0"
    else:
        device = "cpu"
    print(f"      完成，用时 {time.time() - t0:.1f}s")
    print(f"      类别数: {len(model.names)}")
    print(f"      推理设备: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if device == "0" and torch.cuda.is_available() else ""))

    # ---- 2. 打开摄像头 ----
    print(f"[2/3] 打开输入源 {args.source} ...")
    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"      打不开 {args.source}")
        print("      检查：摄像头是否被别的程序占用？索引对不对？（试 --source 1）")
        return 1

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cam = cap.get(cv2.CAP_PROP_FPS)
    print(f"      分辨率 {w}x{h}，摄像头帧率 {fps_cam:.0f}")

    # ---- 3. 主循环 ----
    print("[3/3] 开始检测（按 q / ESC 退出，s 保存截图）")
    print()

    frame_i = 0
    t_start = time.time()
    fps_disp = 0.0
    saved = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("读帧失败，退出")
                break

            # 推理
            results = model.predict(frame, conf=args.conf,
                                    imgsz=args.imgsz, device=device,
                                    verbose=False)
            r = results[0]

            # YOLO 自带的 plot() 会画框 + 标签 + 置信度
            annotated = r.plot()

            # FPS
            frame_i += 1
            elapsed = time.time() - t_start
            if elapsed > 0.5:
                fps_disp = frame_i / elapsed

            # 左上角写 FPS 和物体数
            n_obj = 0 if r.boxes is None else len(r.boxes)
            cv2.putText(annotated, f"FPS {fps_disp:.1f}  obj {n_obj}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2)

            # 每 30 帧打印一次检测结果
            if frame_i % 30 == 0 and r.boxes is not None:
                names = [model.names[int(c)] for c in r.boxes.cls]
                uniq = {}
                for n in names:
                    uniq[n] = uniq.get(n, 0) + 1
                summary = ", ".join(f"{k}x{v}" for k, v in uniq.items())
                print(f"  帧 {frame_i}: {n_obj} 个物体  [{summary}]  FPS {fps_disp:.1f}")

            # 保存截图
            if args.save and not saved:
                out = Path(args.save)
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), annotated)
                print(f"  已保存截图: {out.resolve()}")
                saved = True
                if args.no_display or args.max_frames == 0:
                    break

            # 显示
            if not args.no_display:
                cv2.imshow("YOLO Detection (q/ESC to quit, s to save)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("  用户退出")
                    break
                if key == ord("s"):
                    shot = Path(f"snapshot_{int(time.time())}.jpg")
                    cv2.imwrite(str(shot), annotated)
                    print(f"  已保存 {shot.resolve()}")

            if args.max_frames and frame_i >= args.max_frames:
                print(f"  跑够 {args.max_frames} 帧，退出")
                break

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    total = time.time() - t_start
    print()
    print(f"共处理 {frame_i} 帧，{total:.1f}s，平均 {frame_i/max(total,1e-9):.1f} FPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
