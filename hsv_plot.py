import math
import cv2
import matplotlib.pyplot as plt

samples = [ 'green_normal', 'green_bright', 'red_normal', 'red_bright', 'off_normal', 'off_bright', 'white_normal', 'white_bright' ]

Hx = {}
Hy = {}
for s in samples:
  # 画像を読み込む
  img = cv2.imread('data/samples/' + s + '.png')
  # BGRからHSVに変換する
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  # 色相HをXY座標に変換
  h, w, ch = hsv.shape
  Hx[s] = hx = [ 0.0 ] * (h*w)
  Hy[s] = hy = [ 0.0 ] * (h*w)
  i = 0
  for y in range(h):
    for x in range(w):
      px = hsv[y,x]
      H = 2 * math.pi * px[0] / 180
      S = px[1] / 256
      hx[i] = S * math.sin(H)
      hy[i] = S * math.cos(H)
      i += 1

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
  a.legend()
  # 分布をプロット
  a.scatter(Hx[s], Hy[s], 8)

plt.show()




