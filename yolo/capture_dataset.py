"""
K230 数据集采集脚本
====================
用法：
    1. 把本文件上传到 K230，改名为 main.py 运行
    2. 对准靶框，按下 K230 上的按键（或等定时自动拍）
    3. 照片自动保存到 /sdcard/dataset/
    4. 拍完后，用 mpremote 或拔 SD 卡导出照片到电脑

拍照模式：
    - 按 KEY（BOOT 按键）：手动拍照
    - 或者设置 AUTO_CAPTURE=True，每 N 帧自动拍一张
"""

import time, os, gc
from media.sensor import *
from media.display import *
from media.media import *
from machine import Pin
from machine import FPIOA

# ===================== 用户可调参数 =====================

# 摄像头
CAMERA_ID = 2
IMG_W, IMG_H = 320, 240
IDE_w, IDE_h = 800, 480

# 画面方向（跟比赛保持一致）
SENSOR_HMIRROR = False
SENSOR_VFLIP   = False

# 保存路径（SD 卡）
SAVE_DIR = "/sdcard/dataset"

# 拍照模式
AUTO_CAPTURE = False        # True=自动拍照，False=按键拍照
AUTO_INTERVAL_FRAMES = 60   # 自动模式下每 60 帧拍一张（约 2 秒）

# 按键引脚（立创庐山派 USR 按键 → GPIO53，PULL_DOWN，按下为高电平）
USR_BTN_GPIO = 53
fpioa = FPIOA()
fpioa.set_function(USR_BTN_GPIO, FPIOA.GPIO53)
BUTTON_PIN = Pin(USR_BTN_GPIO, Pin.IN, Pin.PULL_DOWN)

# 显示信息
SHOW_GRID = True  # 是否画辅助网格线，帮助对齐靶框

# ===================== 初始化 =====================

# 创建保存目录（MicroPython 用 os.stat 代替 os.path.exists）
try:
    os.stat(SAVE_DIR)
except OSError:
    os.mkdir(SAVE_DIR)

# 自动编号：找已有最大编号
existing = []
try:
    for f in os.listdir(SAVE_DIR):
        if f.endswith(".jpg"):
            existing.append(f)
except OSError:
    pass

if existing:
    nums = []
    for f in existing:
        try:
            nums.append(int(f.replace(".jpg", "").split("_")[-1]))
        except:
            pass
    counter = max(nums) + 1 if nums else 0
else:
    counter = 0

print(f"保存目录: {SAVE_DIR}")
print(f"起始编号: {counter:04d}")
print(f"拍照模式: {'自动 (每 %d 帧)' % AUTO_INTERVAL_FRAMES if AUTO_CAPTURE else '按键手动'}")

sensor = Sensor(id=CAMERA_ID, width=1280, height=960, fps=90)
sensor.reset()
sensor.set_hmirror(SENSOR_HMIRROR)
sensor.set_vflip(SENSOR_VFLIP)
sensor.set_framesize(width=IMG_W, height=IMG_H)
sensor.set_pixformat(Sensor.RGB888)

Display.init(Display.ST7701, width=IDE_w, height=IDE_h, to_ide=True, quality=70)

# 按键（如果配置了）
if BUTTON_PIN is not None:
    button = BUTTON_PIN
    last_button_state = button.value()
else:
    button = None
    last_button_state = None

MediaManager.init()
sensor.run()

clock = time.clock()
frame_count = 0
saved_count = 0


def save_photo(img, num):
    """保存一张照片"""
    filename = "%s/photo_%04d.jpg" % (SAVE_DIR, num)
    # RGB888 不支持 save → 先转 RGB565 再保存
    img_rgb565 = img.copy(rgb565=True)
    img_rgb565.save(filename, quality=95)
    return filename


def draw_helpers(img):
    """画辅助信息"""
    # 十字线（画面中心）
    cx, cy = IMG_W // 2, IMG_H // 2
    img.draw_line(cx - 20, cy, cx + 20, cy, color=(255, 0, 0), thickness=1)
    img.draw_line(cx, cy - 20, cx, cy + 20, color=(255, 0, 0), thickness=1)

    # 网格线（三分线）
    for i in range(1, 3):
        x = IMG_W * i // 3
        y = IMG_H * i // 3
        img.draw_line(x, 0, x, IMG_H, color=(100, 100, 100), thickness=1)
        img.draw_line(0, y, IMG_W, y, color=(100, 100, 100), thickness=1)


# ===================== 主循环 =====================

print("\n=== 数据采集开始 ===")
if button is None:
    print("按键未配置 → 使用自动拍照模式")
else:
    print("按下 USR 按键拍照")
print("按 Ctrl+C 停止\n")

try:
    while True:
        clock.tick()
        img = sensor.snapshot()

        should_capture = False

        # 自动模式
        if AUTO_CAPTURE and frame_count % AUTO_INTERVAL_FRAMES == 0:
            should_capture = True

        # 按键模式：PULL_DOWN，按下为高电平 1
        if button is not None:
            if button.value() == 1:          # 按下（高电平）
                time.sleep_ms(30)            # 消抖
                if button.value() == 1:      # 确实还在按
                    should_capture = True
                    while button.value() == 1:  # 等松手，防连拍
                        time.sleep_ms(10)

        # 拍照
        if should_capture:
            fname = save_photo(img, counter)
            counter += 1
            saved_count += 1
            print(f"[{saved_count}] 已保存: {fname}")
            # 闪一下画面提示
            img.draw_rectangle(0, 0, IMG_W, IMG_H, color=(0, 255, 0), thickness=3)

        # 画面叠加信息
        if SHOW_GRID:
            draw_helpers(img)

        # 顶部状态栏
        info = "SAVED:%d | #%04d | %.1ffps" % (saved_count, counter, clock.fps())
        mode_str = "AUTO" if AUTO_CAPTURE else "BTN"
        img.draw_string_advanced(5, 5, 10, mode_str + " " + info, color=(255, 255, 255))

        # 底部提示
        if button is not None:
            img.draw_string_advanced(5, IMG_H - 20, 10, "Press USR key", color=(200, 200, 200))

        Display.show_image(img, x=(IDE_w - IMG_W) // 2, y=(IDE_h - IMG_H) // 2)

        frame_count += 1
        if frame_count % 100 == 0:
            gc.collect()

except KeyboardInterrupt:
    pass

finally:
    print(f"\n=== 采集结束，共保存 {saved_count} 张照片 ===")
    print(f"照片路径: {SAVE_DIR}/")
    print(f"拍照张数: {saved_count}")
    print(f"下一个编号: {counter:04d}")
    print("\n导出方法：")
    print("  1. mpremote cp \":/sdcard/dataset/*.jpg\" ./dataset/")
    print("  2. 或者拔 SD 卡，直接用读卡器读到电脑")
    MediaManager.deinit()
