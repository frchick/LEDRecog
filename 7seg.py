import cv2
import numpy as np

# カンマ逆補正LUT
gamma22LUT = np.array([pow(x/255.0 , 2.2) * 255 for x in range(256)], dtype='uint8')

# 認識文字のセグメント発光パターン(1:発光 / 0:消灯)
#  +-0-+
#  1   2
#  +-3-+
#  4   5
#  +-6-+
class char_pattern:
  def __init__(self, ch, pattern):
    self.ch = ch
    self.pattern = pattern

chars = [
  #                  0, 1,2, 3, 4,5, 6
  char_pattern('0', [1, 1,1, 0, 1,1, 1]),
  char_pattern('1', [0, 0,1, 0, 0,1, 0]),
  char_pattern('2', [1, 0,1, 1, 1,0, 1]),
  char_pattern('3', [1, 0,1, 1, 0,1, 1]),
  char_pattern('4', [0, 1,1, 1, 0,1, 0]),
  char_pattern('5', [1, 1,0, 1, 0,1, 1]),
  char_pattern('6', [0, 1,0, 1, 1,1, 1]),
  char_pattern('6', [1, 1,0, 1, 1,1, 1]),
  char_pattern('7', [1, 0,1, 0, 0,1, 0]),
  char_pattern('7', [1, 1,1, 0, 0,1, 0]),
  char_pattern('8', [1, 1,1, 1, 1,1, 1]),
  char_pattern('9', [1, 1,1, 1, 0,1, 0]),
  char_pattern('9', [1, 1,1, 1, 0,1, 1]),

  char_pattern('A', [1, 1,1, 1, 1,1, 0]),
  char_pattern('E', [1, 1,0, 1, 1,0, 1]),
  char_pattern('F', [1, 1,0, 1, 1,0, 0]),
  char_pattern('H', [0, 1,1, 1, 1,1, 0]),
  char_pattern('L', [0, 1,0, 0, 1,0, 1]),
  char_pattern('U', [0, 1,1, 0, 1,1, 1]),

  char_pattern('-', [0, 0,0, 1, 0,0, 0]),
  char_pattern(' ', [0, 0,0, 0, 0,0, 0]),
]

# 7セグマスク画像(グレースケールで読み込む)
img_7seg_mask = []
sum_7seg_mask_pix = []

for i in range(7):
  img = cv2.imread('data/res/seg' + str(i) + '_.png', 0)
  img_7seg_mask.append(img)
  sum_7seg_mask_pix.append(np.sum(img))

# サンプル画像(グレースケールで読み込む)
samples = [ '7seg_F', '7seg_-', '7seg_H', '7seg_1', '7seg_ ' ]
for s in samples:
  # テスト画像を読み込む
  img = cv2.imread('data/samples/' + s + '.png', 0)
  img = cv2.LUT(img, gamma22LUT)
  print(s, end=': ')
  # コントラスト強める
  # ヒストグラムのピークをブラックレベルとする
  hist = cv2.calcHist([img], [0], None, [256], [0, 256])
  black_floor = 0
  hist_max = 0
  for i in range(256):
    if hist_max < hist[i]:
      hist_max = hist[i]
      black_floor = i
  alpha = 2.0 * 255 / (255 - black_floor)
  img = cv2.convertScaleAbs(img, alpha=alpha, beta=-alpha*black_floor)
  # セグメント毎の、正規化された輝度を計算
  separator = ''
  v = [ 0.0 ] * 7
  for i in range(7):
    masked_img = cv2.multiply(img, img_7seg_mask[i], scale=1.0/255.0)
    sum_pix = np.sum(masked_img)
    v[i] = sum_pix / sum_7seg_mask_pix[i]
    print(separator + '{:.2f}'.format(v[i]), end='')
    separator = ', '

  # どの文字かを判定する
  max_vv = 0.0
  ch = '?'
  for c in chars:
    vv = 0
    for i in range(7):
      if 0 < c.pattern[i]:
        vv += v[i]
      else:
        vv += (1.0 - v[i])
    if max_vv < vv:
      max_vv = vv
      ch = c.ch
  print(': ch=' + ch)
