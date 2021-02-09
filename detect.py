# -*- coding: utf-8 -*-
#===============================NNの定義=======================================
Classification_Num = 3
n_in = 8
n_mid = 50
n_out = Classification_Num
wb_width = 0.1
eta = 0.01

# -- 各層の継承元 --
class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = wb_width * np.random.randn(n_upper, n)  # 重み（行列）
        self.b = wb_width * np.random.randn(n)  # バイアス（ベクトル）

        self.h_w = np.zeros(( n_upper, n)) + 1e-8
        self.h_b = np.zeros(n) + 1e-8
        
    def update(self, eta):      
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        
        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b

# -- 中間層 --
class MiddleLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u)  # ReLU
    
    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)  # ReLUの微分

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 

# -- 出力層 --
class OutputLayer(BaseLayer):     
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u)/np.sum(np.exp(u), axis=1, keepdims=True)  # ソフトマックス関数

    def backward(self, t):
        delta = self.y - t
        
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 
        
# -- ドロップアプト --
class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio  # ニューロンを無効にする確率

    def forward(self, x, is_train):  # is_train: 学習時はTrue
        if is_train:
            rand = np.random.rand(*x.shape)  # 入力と同じ形状の乱数の行列
            self.dropout = np.where(rand > self.dropout_ratio, 1, 0)  # 1:有効 0:無効
            self.y = x * self.dropout  # ニューロンをランダムに無効化
        else:
            self.y = (1-self.dropout_ratio)*x  # テスト時は出力を下げる
        
    def backward(self, grad_y):
        self.grad_x = grad_y * self.dropout  # 無効なニューロンでは逆伝播しない

# -- 順伝播 --
def fp(x, is_train):
    ml_1.forward(x)
    dp_1.forward(ml_1.y, is_train)
    ml_2.forward(dp_1.y)
    dp_2.forward(ml_2.y, is_train)
    ol.forward(dp_2.y)

#===================================================================================


# csvを読んで要素をintにする
def str2int(path):
    with open(path) as f:
        reader = csv.reader(f)
        data = np.array([row for row in reader]).astype(np.int64)
    return data




if __name__ == '__main__':
    rospy.init_node('NN')
    # -- 各層の初期化 --
    ml_1 = MiddleLayer(n_in, n_mid)
    dp_1 = Dropout(0.5)
    ml_2 = MiddleLayer(n_mid, n_mid)
    dp_2 = Dropout(0.5)
    ol = OutputLayer(n_mid, n_out)


    #学習済みの重みを読み込む
    w_ml_1 = dir_here + '/csv/NN-result/w_ml_1.csv'
    b_ml_1 = dir_here + '/csv/NN-result/b_ml_1.csv'
    w_ml_2 = dir_here + '/csv/NN-result/w_ml_2.csv'
    b_ml_2 = dir_here + '/csv/NN-result/b_ml_2.csv'
    w_ol = dir_here + '/csv/NN-result/w_ol.csv'
    b_ol = dir_here + '/csv/NN-result/b_ol.csv'
        
    ml_1.w = str2int(w_ml_1)[0]
    ml_1.b = str2int(b_ml_1)[0]
    ml_2.w = str2int(w_ml_2)[0]
    ml_2.b = str2int(b_ml_2)[0]
    ol.w = str2int(w_ol)[0]
    ol.b = str2int(b_ol)[0]

    fname = '/home/naoyamada/catkin_ws3/src/hsr_handing/script/csv/rt-pose.csv'
    pub = rospy.Publisher('pose_topic', pose_info, queue_size=10)
    while not rospy.is_shutdown():


	    fp(input_data, is_train=False)
	    index_result = np.argmax(ol.y,axis=1)
	    probability = round(float(ol.y[0][index_result])*100,3)
	    pose_identification_number=3
	    if index_result==0 and probability>=50:
	        print('座り---{}%'.format(probability))
	        pose_identification_number = index_result
	    if index_result==1 and probability>=50:
	        print('立ち---{}%'.format(probability))
	        pose_identification_number = index_result
	    if index_result==2 and probability>=50:
	        print('あぐら---{}%'.format(probability))  
	        pose_identification_number = index_result 
	    if probability<50 :
	        pose_identification_number = 3

        pose_identification_number = 0
        pub.publish(pose_identification_number)