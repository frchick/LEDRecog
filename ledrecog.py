import cv2
from matplotlib import pyplot as plt

samples = [ 'green_normal', 'green_bright', 'red_normal', 'red_bright', 'off_normal', 'off_bright' ]
img = {}
hsv = {}
h_hist = {}
s_hist = {}
v_hist = {}
for s in samples:
  # 画像を読み込む
  img[s] = cv2.imread('data/samples/' + s + '.png')
  # BGRからHSVに変換する
  hsv[s] = cv2.cvtColor(img[s], cv2.COLOR_BGR2HSV)
  # H、S、Vの分布を計算する
  h_hist[s] = cv2.calcHist([hsv[s]],[0], None, [180], [0,180])
  s_hist[s] = cv2.calcHist([hsv[s]],[1], None, [256], [0,256])
  v_hist[s] = cv2.calcHist([hsv[s]],[2], None, [256], [0,256])

# サブプロットを作成する
fig, ax = plt.subplots(2, 3)

# 分布をプロットする
for i, s in enumerate(samples):
  col = i // 2
  row = i % 2
  ax[row, col].set_title(s)
  ax[row, col].plot(h_hist[s],label='Hue')
  ax[row, col].plot(s_hist[s],label='Saturation')
  ax[row, col].plot(v_hist[s],label='Value')
  ax[row, col].legend()

plt.show()
