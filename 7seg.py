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

# テストmain関数
def main():
  # サンプル画像(グレースケールで読み込む)
  samples = [ '7seg_F', '7seg_-', '7seg_H', '7seg_1', '7seg_ ' ]
  for s in samples:
    # テスト画像を読み込む
    img = cv2.imread('data/samples/' + s + '.png', cv2.IMREAD_GRAYSCALE)
    print(s, end=': ')
    # 文字の読み取り
    ch = read_7seg_char(img)
    print(' > ch=' + ch)

img_7seg_mask = []

# 7セグマスク画像を読み込む
def load_7seg_masks():
  # すでに読み込み済みなら何もしない
  if 0 == len(img_7seg_mask):
    # 7セグマスク画像(グレースケールで読み込む)
    for i in range(7):
      img = cv2.imread('data/res/seg' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
      img = cv2.LUT(img, gamma22LUT)
      img_7seg_mask.append(img)

# 7セグ文字の読み取り
def read_7seg_char(img):
  # 7セグマスク画像を読み込む
  load_7seg_masks()
  # 輝度をリニア化
  img = cv2.LUT(img, gamma22LUT)
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
  # セグメント毎にマスクされた輝度と、その反転輝度を計算
  v = [ 0.0 ] * 7
  iv = [ 0.0 ] * 7
  inv_img = cv2.bitwise_not(img)
  for i in range(7):
    masked_img = cv2.multiply(img, img_7seg_mask[i], scale=1.0/255.0)
    v[i] = np.sum(masked_img)
    masked_img = cv2.multiply(inv_img, img_7seg_mask[i], scale=1.0/255.0)
    iv[i] = np.sum(masked_img)

  # どの文字かを判定する
  max_vv = 0.0
  ch = '?'
  separator = ''
  for c in chars:
    vv = 0
    for i in range(7):
      if 0 < c.pattern[i]:
        vv += v[i]
      else:
        vv += iv[i]
    if max_vv < vv:
      max_vv = vv
      ch = c.ch
    print(separator + c.ch + ':' + str(vv), end='')
    separator = ', '
  return ch

# エントリーポイント
if __name__ == "__main__":
  main()