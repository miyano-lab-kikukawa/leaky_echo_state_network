import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA


# 関数定義--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def activation_function(x):                     # 活性化関数の定義
    return np.tanh(x) 

def echo_state_property(weights):
    values, vectors = np.linalg.eig(weights)    # np.linalg.eig は valuesが固有値を含んだ配列、vectordが固有ベクトルを求める操作

    spectral_radius = max(abs(values))          # 固有値から最大の絶対値を求める
    W_res = weights / spectral_radius           # 各ノードの重さを固有値の最大値で割ることで、各重さパラメータを正規化する

    return W_res

def ridge_regression(N, xs, ys, alpha):         # リッジ回帰では、損失関数の値を最小化するとき、重みパラメータの値が極端に大きな数値で最適化され複雑な関数(過学習状態）とならないために正則化項を加える
                                                # 即ち、リッジ回帰とは、重みパラメータの各値が大きな値にならない範囲で損失関数が最も小さくなるように最適な重みパラメータを探す操作
    regularization_term = np.identity(N) * alpha
    xs1 = np.linalg.inv(np.dot(xs, xs.T) + regularization_term)
    save_xs = np.dot(xs1, xs)
    return np.dot(ys, save_xs.T)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Leaky_echo_state_network:

    def __init__(self, input_data, input_node_num, reservoir_node_num, output_node_num, learning_rate, training_percent=0.8, leaky_rate=0.1):    # 初期情報
        self.input_data = input_data
        self.input_data_size = self.input_data.size

        self.input_node_num = input_node_num                                                                                          # input層のノード数
        self.reservoir_node_num = reservoir_node_num                                                                                  # reservoir層のノード数
        self.output_node_num = output_node_num                                                                                        # output層のノード数

        self.learning_rate = learning_rate                                                                                            # 学習率
        self.training_percent = training_percent                                                                                      # 学習範囲の指定
        self.leaky_rate = leaky_rate                                                                                                  # 漏れ率

        self.training_data_size = int(self.input_data_size * self.training_percent)                                                   # trainig_dataの作成
        self.test_data_size = self.input_data_size - self.training_data_size                                                          # test_dataの作成

        self.training_data, self.test_data = np.hsplit(self.input_data, [self.training_data_size])                                    # trainig_dataとtest_dataをtrainig_dataの値の最後の位置で分割 np.hsplit() = np.split(axis=1)

        self.W_in = np.random.choice([-0.1,0.1], input_node_num*reservoir_node_num).reshape(reservoir_node_num, input_node_num)       # -0.1or0.1の一様分布となるW_inの作成
        self.W_res = self.__make_reservoir_weights()                                                                                  # echo_state_propertyを満たすW_resの作成
        self.W_out = np.zeros([output_node_num, reservoir_node_num])                                                                  # 出力W_outの初期値

        self.x = np.zeros((reservoir_node_num, 1))                                                                                    # 内部パラメータを要素'0'で初期化

        self.x_repository = self.x                                                                                                    # 内部状態xの保存用 update_output_weightsの更新時に使用する                                                                                      

        self.time_data = np.linspace(0, self.input_data_size-1, self.input_data_size)                                                 # 全体の時間データ
        self.training_time_data = self.time_data[0:self.training_data_size+1]                                                         # 学習の時間データ
        self.test_time_data = self.time_data[self.training_data_size:]                                                                # テストの時間データ

        self.output_data = None                                                                                                       # 出力データ
        self.trained_data = None                                                                                                      # 学習済みデータ    

    def __make_reservoir_weights(self, percent=0.8):                                      # percent:ノード間の結合率
        N = self.reservoir_node_num
        zero_num = int((N**2)*percent)                                                    # 0の配列を生成（全体の8割）
        one_num = int((N**2)-zero_num)
        zero_and_one = np.concatenate([np.zeros(zero_num), np.ones(one_num)])             # 1の配列を生成（全体の2割）し、0の配列と結合する
        weights = np.random.permutation(zero_and_one).reshape([N, N])                     # 生成した配列（'0'要素8割、'1'要素2割をランダムに並び替え、要素N×Nの配列を作成

        W_res = echo_state_property(weights)                                              # echo state propertyの条件処理
        return W_res

    def __internal_function(self, u, x):                                                                                              # reservoir層の内部状態の更新
        x = activation_function((1 - self.leaky_rate) * x + self.leaky_rate * (np.dot(self.W_in, u) + np.dot(self.W_res, x)))         # 内部状態の更新式
        return x

    def run_network(self):                                                                # 学習用演算
        for i in range(self.training_data_size):
            u = self.training_data[i]                                                     # 入力データの取り込み
            x = self.x
            self.x = self.__internal_function(u, x)                                       # 内部状態の更新計算
            self.x_repository = np.concatenate([self.x_repository, self.x], 1)            # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0].T ← もともとのデータ（1列目,i=0）

    def update_output_weights(self):                                                                                                  # リッジ回帰を用いたoutput層のweightsの更新
        self.x_repository = np.delete(self.x_repository, -1, 1)                                                                       # 一番最後の列の値を消去
        self.W_out = ridge_regression(self.reservoir_node_num, self.x_repository, self.training_data, self.learning_rate)             # リッジ回帰を用いてW_outの値を更新

    def check_trained(self):                                                              # 学習データの予測結果
        x = np.zeros((self.reservoir_node_num, 1))                                        # 初期内部状態が0になるように設定
        self.trained_data = self.training_data[0]                                         # 学習データの初期値設定

        for i in range(self.training_data_size):                                          # 学習領域（全体の8割）まで
            u = self.training_data[i]                                                     # 学習データの入力
            x = self.__internal_function(u, x)                                            # 内部状態の計算
            z = np.dot(self.W_out, x)                                                     # 上記の内部状態における出力値の計算

            self.trained_data = np.append(self.trained_data, z)                           # 得られた出力値データの保存

    def predict(self):                                                                    # 予測
        self.output_data = np.dot(self.W_out, self.x)                                     # 出力データの初期値として学習データの最後の値を使用する

        for i in range(self.test_data_size - 1):                                          # 学習外の領域を予測する（全体の2割） 
            u = self.output_data[i]                                                       # 入力データに出力データの最後の値を使用する（初期値は学習のデータの最終値）
            self.x = self.__internal_function(u, self.x)                                  # 内部状態の更新
            z = np.dot(self.W_out, self.x)                                                # 予測した出力データ

            self.output_data = np.append(self.output_data, z)                             # 予測した出力データの保存

    def graph(self):                                                                      # グラフ作成
        fig = plt.figure(figsize=(15, 8))                                                 # plt.figure()のfigsizeは(width, height)のタプルを渡し、単位はインチ
        ax = fig.add_subplot(111)                                                         # グラフの描写領域設定 (111)は１行目１列１番目という意味で、subplot(1,1,1)も同じ
        ax.set_title('Leaky Echo State Network', fontsize=25)                             # サブプロットaxのタイトル設定
        ax.set_xlabel('Time step', fontsize=23)                                           # axのx軸タイトル設定
        ax.set_ylabel('Amplitude', fontsize=23)                                           # axのy軸タイトル設定
        plt.tick_params(labelsize=20)                                                     # グラフのパラメータ設定からラベルの文字の大きさを設定

        def __graph(time, data, name, color):                                             # グラフ作成の定義
            plt.plot(time, data, linewidth=2, label=name, color=color)

        __graph(self.time_data, self.input_data, "input", 'blue')                         # 入力関数を表示

        if self.trained_data is not None:                                                 # 学習データ関数を表示する時の条件付け
            __graph(self.training_time_data, self.trained_data, "trained", 'green')       # 学習データ関数の表示

        if self.test_data is not None:                                                    # 予測データ関数を表示する時の条件付け
            __graph(self.test_time_data, self.output_data, "predict", 'red')              # 予測データ関数の表示

        plt.legend()                                                                      # グラフに判例を表示する
        plt.grid()                                                                        # グラフにグリッドを表示する
        plt.show()                                                                        # グラフをウィンドウに出力する
        plt.close()                                                                       # 出力されたグラフウィンドウを閉じる

    def __repr__(self):                                                                   # クラスの名前付け
        return 'Leaky Echo State Network'                                                 # a = Leaky_echo_state_network() とした時に print(a)で表示される文字列

if __name__ == '__main__':                                                                # このファイルがコマンドラインからスクリプトとして実行された場合にのみ以降の処理を実行する
                                                                                          # 他のファイルからインポートされたときは実行されない

    t = np.linspace(0, 40*np.pi, int((40*np.pi)/(np.pi*0.01))+1)                          # 等差数列を作成 np.linspace(始点,　終点,　サンプル数)
    u = np.sin(t) * 1                                                                     # 振幅1のsin波

    network = Leaky_echo_state_network(u, 1, 20, 1, 0.6)
    network.run_network()
    network.update_output_weights()
    network.check_trained()
    network.predict()
    network.graph()
