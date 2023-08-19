import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft import fft
from wave_io import read
from window import * # 4回目講義で作成した窓関数をインポートして使用


# 短時間フーリエ変換（short-time fourier transform; STFT）を実行する
#   - x: 入力信号
#   - w: 窓関数（窓幅: M）
#   - frame_shift: 何サンプルおきにDFTを実行するか
def STFT(x: np.ndarray, w: np.ndarray, frame_shift: int):

    N = len(x) # 入力信号の長さ
    M = len(w) # 窓関数の長さ

    X = []
    for n in range(0, N, frame_shift):

        # 【TODO】
        # 長さ M の一次元配列 y を用意


        # 【TODO】
        # x[n] 〜 x[n+M-1] に窓関数 w を適用し，結果を y に保存


        # 【TODO】
        # y に対しDFTを実行し，結果を配列 a に保存


        # DFT結果を出力変数にコピー
        X.append(a)

    return np.asarray(X)


# スペクトログラムを求めて表示する
#   - x: 入力信号
#   - w: 窓関数（窓幅: M）
#   - frame_shift: 何サンプルおきにDFTを実行するか
#   - fu: 表示上の上限となる周波数（デフォルトでは 22050 [Hz]）
#   - fs: サンプリング周波数（本講義では 44100 [Hz]で固定とする）
def show_spectrogram(x: np.ndarray, w: np.ndarray, frame_shift=512, fu=20500, fs=44100):

    N = len(x) # 入力信号の長さ
    M = len(w) # 窓関数の長さ

    # STFTを実行
    X = STFT(x, w, frame_shift=frame_shift)

    # 【TODO】
    # DFT係数の絶対値（振幅）のみを取り出して配列 X_abs に保存


    # 返値として使用する情報を退避
    result = X_abs[:, :M//2+1] # スペクトログラムの生データ
    phase = np.angle(X)[:, :M//2+1] # 位相情報

    # 0 ～ fu [Hz] に相当する範囲のSTFT係数のみを残し，残りはカット
    fu = min(fu, fs / 2)
    Mu = int(M * fu / fs) + 1
    X_abs = X_abs[:, :Mu]

    # 見やすくするための調整
    Mu = 512
    Nu = 1600
    X_abs = X_abs.T # 縦軸が周波数軸，横軸が時間軸となるようにする
    X_abs = np.log(1 + X_abs) # 振幅の強さを log スケールで表すようにする
    X_abs = np.asarray(Image.fromarray(X_abs).resize((Nu, Mu), Image.Resampling.NEAREST)) # 表示用画像のサイズを調整

    # 縦軸（周波数）・横軸（時間）の目盛りを設定
    L = N / fs
    x_step = 1 if L > 2 else 0.25 # 横軸は　0.25[秒] または 1.0[秒] 刻み
    y_step = 2000 if fu > 10000 else 1000 # 縦軸は 1000[Hz] または 2000[Hz] 刻み
    x_scale = np.arange(0, L, x_step)
    y_scale = np.arange(0, fu, y_step)
    x_pos0 = np.asarray([round(i * fs) for i in x_scale])
    x_pos1 = np.asarray([round(i * Nu / L) for i in x_scale])
    y_pos = np.asarray([round(i * Mu / fu) for i in y_scale])

    # スペクトログラムを表示
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), tight_layout=True, gridspec_kw=dict(height_ratios=[3, 7]))
    ax[0].plot(np.arange(0, N), x, linewidth=0.1)
    ax[0].set_xlim([0, N])
    ax[0].set_xticks(x_pos0)
    ax[0].set_xticklabels(x_scale)
    ax[0].set_title('waveform')
    ax[1].imshow(X_abs)
    ax[1].set_xticks(x_pos1)
    ax[1].set_yticks(y_pos)
    ax[1].set_xticklabels(x_scale)
    ax[1].set_yticklabels(y_scale)
    ax[1].invert_yaxis()
    ax[1].set_xlabel('time [sec.]')
    ax[1].set_ylabel('frequency [Hz]')
    ax[1].set_title('spectrogram')
    plt.show()

    return result, phase


# エントリポイント： C言語で言うところの main 関数のようなもの（本当は違うが，そういう理解で十分）
if __name__ == "__main__":

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'Spectrogram viewer')
    parser.add_argument('--in_file', '-i', required=True, type=str, help='input filename')
    parser.add_argument('--out_file', '-o', type=str, default='result.pkl', help='output data filename')
    parser.add_argument('--window_type', '-t', default='rect', type=str, help='type of window')
    parser.add_argument('--frame_shift', '-s', default=256, type=int, help='frame shift for STFT')
    parser.add_argument('--frame_size', '-z', default=1024, type=int, help='frame size (length of window)')
    parser.add_argument('--max_frequency', '-f', default=22050, type=int, help='maximum frequency for displaying spectrogram')
    args = parser.parse_args()

    # コマンドライン引数で指定したパラメータやファイル名を変数に格納
    input_filename = args.in_file # 入力ファイル名（文字列型）
    output_filename = args.out_file # 出力ファイル名（文字列型）
    window_type = args.window_type # 窓関数の種類
    frame_shift = args.frame_shift # STFTにおけるフレームシフト
    frame_size = args.frame_size # STFTにおけるフレームサイズ
    fu = args.max_frequency # 表示上の上限周波数

    # 窓関数を作成
    M = frame_size # 窓関数の長さ == フレームサイズ
    if window_type == 'rect':
        w = make_rect_window(M)
    elif window_type == 'han':
        w = make_han_window(M)
    elif window_type == 'hamming':
        w = make_hamming_window(M)
    elif window_type == 'blackman':
        w = make_blackman_window(M)
    else:
        print('error: select one of the following window types: rect, han, hamming, and blackman')
        exit()

    # 入力ファイルから音響信号を読み出し，一次元配列 x に格納
    x = read(input_filename)

    # スペクトログラムを計算・表示
    X, phase = show_spectrogram(x, w, frame_shift, fu)

    # 計算結果をファイルに保存
    with open(output_filename, 'wb') as f:
        pickle.dump((X, w, frame_shift, phase), f)
