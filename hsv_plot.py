import math
import cv2
import matplotlib.pyplot as plt
import numpy

def clamp(v, min_v, max_v):
    return max(min_v, min(v, max_v))

# サンプル画像
samples = [ 'green_normal', 'green_bright', 'red_normal', 'red_bright', 'off_normal', 'off_bright', 'white_normal', 'white_bright' ]
#samples = [ 'green', 'red', 'off', 'white' ]

# カンマ逆補正LUT
gamma22LUT = numpy.array([pow(x/255.0 , 2.2) * 255 for x in range(256)], dtype='uint8')


ImgGray = {}
Hx = {}
Hy = {}
Col = {}
w = 0
h = 0
for s in samples:
  # 画像を読み込む
  img = cv2.imread('data/samples/' + s + '.png')
  # BGRからHSVに変換する
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
  # 色相HをXY座標に変換
  h, w, ch = hsv.shape
  Hx[s] = hx = [ 0.0 ] * (h*w)    # 色相環のX座標(-1.0 <= Hx <= 1.0)
  Hy[s] = hy = [ 0.0 ] * (h*w)    # 色相環のY座標(-1.0 <= Hy <= 1.0)
  Col[s] = col = [ '' ] * (h*w)   # 無彩色(k,w) or カラー判定(r,g,b)
  i = 0
  for y in range(h):
    for x in range(w):
      px = hsv[y,x]
      H = 2 * math.pi * px[0] / 256   # 色相(RAD)
      Hd = 360 * px[0] / 256          # 色相(DEG)
      S = px[1] / 255                 # 彩度
      V = px[2] / 255                 # 明度
      hx[i] = S * math.sin(H)
      hy[i] = S * math.cos(H)
      # プロットカラー
      if V < 0.50:    # 暗いピクセルは黒
        col[i] = 'k'
      elif S < 0.25:  # グレー～白のピクセルは白
        col[i] = 'w'
      else:           # あとはRGB
        if (60 < Hd) and (Hd <= 180):
          col[i] = 'g'
        elif (180 < Hd) and (Hd <= 300):
          col[i] = 'b'
        else:
          col[i] = 'r'
      i += 1
  # グレースケール画像を作成
  ImgGray[s] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 点灯消灯を判定するカーネルを作成
mask = cv2.imread('data/res/circle_mask.png', 0)
cntp = 0
cntn = 0
kernel = [[]] * h
for y in range(h):
  kernel[y] = [ 0.0 ] * w
  for x in range(w):
    kernel[y][x] = k = clamp((mask[y,x] - 128) / 127, -1.0, 1.0)
    if 0.0 < k:
      cntp += 1
    elif k < 0.0:
      cntn += 1
for y in range(h):
  for x in range(w):
    if 0.0 < kernel[y][x]:
      kernel[y][x] /= cntp
    elif kernel[y][x] < 0.0:
      kernel[y][x] /= cntn

# 点灯消灯を判定
conv = {}
for s in samples:
  c = 0.0
  img = ImgGray[s]
  for y in range(h):
    for x in range(w):
      c += kernel[y][x] * img[y,x]
  conv[s] = c

# サブプロットを作成する
fig, ax = plt.subplots(2, 4)
# 色相環の画像
hsvCircleImg = plt.imread('data/res/hsv_circle.png')

# 分布をプロットする
for i, s in enumerate(samples):
  # グラフを設定
  col = i // 2
  row = i % 2
  a = ax[row, col]
  a.set_xlim(-1.0, 1.0)
  a.set_ylim(-1.0, 1.0)
  a.imshow(hsvCircleImg, extent=[-1.0, 1.0, -1.0, 1.0], zorder=-1)
  a.set_title(s)
  a.set_xticks([])
  a.set_yticks([])
#  a.set_title(s + ' / conv=' + str(int(conv[s])))
#  a.legend()
  # 分布をプロット
  a.scatter(Hx[s], Hy[s], 8, color=Col[s])

plt.show()




