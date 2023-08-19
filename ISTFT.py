import pickle
import argparse
import numpy as np
from numpy.fft import ifft
from wave_io import save
from spectrogram import STFT


# 逆STFTによりスペクトログラム（の生データ）から音響信号を復元する
#   - X: スペクトログラムの生データ
#   - w: スペクトログラムの計算に用いたものと同じ窓関数
#   - frame_shift: スペクトログラムの計算に用いたものと同じフレームシフト値
#   - phase: 位相情報（ None の場合は 0～2π の一様乱数を用いて適当に設定）
def ISTFT(X: np.ndarray, w: np.ndarray, frame_shift: int, phase=None):

    # 位相情報が存在しない場合は乱数で適当に設定
    if phase is None:
        phase = 2 * np.pi * np.random.rand(*X.shape)

    # 音響信号を復元（細部の説明は省略）
    N = len(X)
    M = len(w)
    x = np.zeros((N-1) * frame_shift + M)
    z = np.zeros((N-1) * frame_shift + M)
    for i in range(0, N):
        a = np.concatenate([X[i], X[i, 1:-1][::-1]], axis=0)
        p = np.concatenate([phase[i], -phase[i, 1:-1][::-1]], axis=0)
        y = ifft(a * np.exp(1j * p))
        x[i*frame_shift:i*frame_shift+M] += w*y.real
        z[i*frame_shift:i*frame_shift+M] += w**2
    z[np.where(z == 0)[0]] = 1
    x /= z

    return x


# 反復位相推定に基づく逆STFT
#   - X: スペクトログラムの生データ
#   - w: スペクトログラムの計算に用いたものと同じ窓関数
#   - frame_shift: スペクトログラムの計算に用いたものと同じフレームシフト値
#   - n_iter: 反復回数
def ISTFT_with_phase_estimation(X: np.ndarray, w: np.ndarray, frame_shift: int, n_iter: int):

    M = len(w)

    x = ISTFT(X, w, frame_shift, None)
    for i in range(n_iter - 1):
        phase = np.angle(STFT(x, w, frame_shift)[:, :M//2+1])
        x = ISTFT(X, w, frame_shift, phase)

    return x


# エントリポイント： C言語で言うところの main 関数のようなもの（本当は違うが，そういう理解で十分）
if __name__ == "__main__":

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'Sound reconstructor from a spectrogram')
    parser.add_argument('--in_file', '-i', default='result.pkl', type=str, help='input filename (result file made by spectrogram.py)')
    parser.add_argument('--out_file', '-o', type=str, default='ISTFT.wav', help='output sound filename')
    parser.add_argument('--n_iterations', '-n', default=1, type=int, help='num. iterations for phase estimation')
    args = parser.parse_args()

    # コマンドライン引数で指定したパラメータ値やファイル名を変数に格納
    input_filename = args.in_file # 入力ファイル名（文字列型）
    output_filename = args.out_file # 出力ファイル名（文字列型）
    n_iter = args.n_iterations

    # 入力ファイルからスペクトログラムの情報を取得
    with open(input_filename, 'rb') as f:
        X, w, frame_shift, phase = pickle.load(f)

    # スペクトログラムから音響信号を復元し，一次元配列 x に格納
    # あえて正しい位相情報を捨て，ランダムに設定した位相から復元してみる
    if n_iter < 2:
        x = ISTFT(X, w, frame_shift, phase=None)
    else:
        x = ISTFT_with_phase_estimation(X, w, frame_shift, n_iter)

    # 復元した信号をファイルに保存
    save(x, output_filename)
