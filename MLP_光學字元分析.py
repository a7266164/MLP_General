'''
本程式以MNIST的部份手寫數字0-9的照片訓練MLP模型，並在訓練完後以另一部份照片測試分類性能。
'''

import os
import matplotlib.pyplot as plt
import json
import time
import torch
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from hyperopt import fmin, tpe, hp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold, train_test_split

#========================================輔助函數========================================#

# 生成檔名和儲存路徑
def generate_filename(base_path, stage, model_num, content, file_type):
    if stage == "find_hyper":
        directory = f"{base_path}/best_hyperparameter"
        save_path = f"{directory}/{content}.{file_type}"
    elif stage == "single_result":
        directory = f"{base_path}/single_model_result"
        save_path = f"{directory}/{content}_{model_num}.{file_type}"
    elif stage == "total_result":
        directory = f"{base_path}/total_result"
        save_path = f"{directory}/{content}.{file_type}"
    else:
        raise ValueError("Invalid stage value")

    # 創建目錄，如果它不存在
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    return save_path

#========================================輔助函數========================================#





#=========================================主函數=========================================#

# 資料預處理
def data_preprocess():
    
    """
    進行資料預處理和超參數讀取。
    
    輸入:
    - dataset_path: 資料集路徑

    輸出:
    - t_dataset (TensorDataset): 包含特徵和標籤的 PyTorch TensorDataset。
    """

    # 定義數據預處理步驟
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST 數據集的均值和標準差
    ])

    # 下載訓練集和測試集
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
    return train_dataset, test_dataset

# 模型定義
class MLP(nn.Module):
    """
    定義多層感知器(MLP)。
    """
    def __init__(self, input_dim, output_dim, total_hidden_layers, neurons_per_layer, dropout_rate):
        """
        初始化 MLP。
        
        參數:
        - input_dim (int): 輸入維度。
        - output_dim (int): 輸出維度。
        - total_hidden_layers (int): 隱藏層的總數量。
        - neurons_per_layer (int): 每個隱藏層的神經元數量。
        - dropout_rate (float): 丟棄比例。
        """
        super(MLP, self).__init__()

        # 創建神經元層（包括輸入、隱藏和輸出層）
        self.create_layers(input_dim, output_dim, total_hidden_layers, neurons_per_layer)

        # 設定丟棄率
        self.dropout_rate = dropout_rate

    def create_layers(self, input_dim, output_dim, total_hidden_layers, neurons_per_layer):
        """創建神經元層"""
        # 初始化 layers
        self.layers = nn.ModuleList()
        
        # 創建一個列表，其中包含 total_hidden_layers 個元素，每個元素的值都是 neurons_per_layer
        total_hidden_layers = int(total_hidden_layers)
        neurons_per_layer = int(neurons_per_layer)
        hidden_dims = [neurons_per_layer] * total_hidden_layers

        # 創建輸入層、隱藏層
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(last_dim, hidden_dim))
            last_dim = hidden_dim

        # 創建輸出層
        self.output_layer = nn.Linear(last_dim, output_dim)
        
    def forward(self, x):
        """
        前向傳播函數。
        
        參數:
        - x (Tensor): 輸入數據。
        
        返回:
        - x (Tensor): 輸出數據。
        """
        activation_function = nn.LeakyReLU(0.02)
        dropout = nn.Dropout(self.dropout_rate)
       
        # 調整資料形狀
        x = x.view(x.shape[0], -1)
        
        # 通過隱藏層
        for layer in self.layers:
            x = activation_function(layer(x))  # 隱藏層使用Leaky Relu激活函數
            x = dropout(x)  # 通過時進行 dropout
        
        x = self.output_layer(x)
        return x

# 找出最優超參數
def hyperparameter_best(t_dataset,
                        input_dim, 
                        output_dim,
                        max_evals, 
                        hidden_layers_choices,
                        neurons_per_layer_choices,
                        dropout_rate_range,
                        learning_rate_range,
                        optimizer_choices,
                        k_folds,
                        early_stopping                      
                        ):

    """
    尋找一組最優超參數。
    
    輸入:
    - dataset (TensorDataset)   : 包含特徵和標籤的 PyTorch TensorDataset。
    - input_dim                 : 輸入維度。
    - output_dim                : 輸出維度。
    - max_evals                 : 貝葉斯搜索次數。
    - hidden_layers_choices     : 隱藏層區間，為一元組，內容為(下限, 上限, 步階)。
    - neurons_per_layer_choices : 每層神經元區間，為一元組，內容為(下限, 上限, 步階)。
    - dropout_rate_range        : 神經元拔除區間，為一元組，內容為(下限, 上限)。
    - learning_rate_range       : 學習區間，為一元組，內容為(下限, 上限)。
    - optimizer_choices         : 優化器選項，為一列表，內容為['選項一', '選項二', '選項三'...(可自行添加)]。
    
    輸出:
    - best_params               : 一組範圍內最佳的超參數組合。
    """

    # 檢測是否有可用的GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 印出使用的裝置
    print(f"Using device: {device}")

    # 如果是使用 CPU，則暫停程式並等待用戶確認
    if device.type == 'cpu':
        input("No GPU found. Press any key to continue...")

    # 設定超參數範圍
    param_grid = {
        'total_hidden_layer': hp.quniform   ('total_hidden_layer', hidden_layers_choices[0], hidden_layers_choices[1], hidden_layers_choices[2]),
        'neurons_per_layer' : hp.quniform   ('neurons_per_layer', neurons_per_layer_choices[0], neurons_per_layer_choices[1], neurons_per_layer_choices[2]),
        'dropout_rate'      : hp.uniform    ('dropout_rate', *dropout_rate_range),
        'learning_rate'     : hp.uniform    ('learning_rate', *learning_rate_range),
        'optimizer'         : hp.choice     ('optimizer', optimizer_choices)
    }

    # 初始化最佳參數的字典
    best_params = {}  

    # 目標函數
    def objective(params):
        """
        目標函數。
        
        輸入:
        - params : 該次蒐索的組合
        
        輸出:
        - np.mean(k_fold_cross_entropy_loss) : 該次蒐索的K折平均損失。
        """
        # 儲存K折交叉熵損失
        k_fold_cross_entropy_loss = []  

        # K折
        kf = KFold(n_splits=k_folds)
        
        # 主迴圈
        for train_index, test_index in kf.split(t_dataset):
            
            #=========================================初始化=========================================#
            
            # 早停計數器
            early_stopping_counter = 0

            # 最佳交叉熵設為無窮大
            best_cross_entropy_loss = np.inf  
            
            #=========================================初始化=========================================#





            #=======================================程式運行區=======================================#
            
            # 分割子集
            train_dataset = torch.utils.data.Subset(t_dataset, train_index)
            test_dataset = torch.utils.data.Subset(t_dataset, test_index)
            
            # 創建資料加載器
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
            testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            # 讀取超參數
            total_hidden_layers = params['total_hidden_layer']
            neurons_per_layer = params['neurons_per_layer']
            dropout_rate = params['dropout_rate']
            learning_rate = params['learning_rate']
            optimizer_type = params['optimizer']

            # 創建模型
            model = MLP(input_dim, output_dim, total_hidden_layers, neurons_per_layer, dropout_rate).to(device)
        
            # 損失函數：二元交叉熵
            criterion = nn.CrossEntropyLoss()

            if optimizer_type == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_type == 'sgd':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            elif optimizer_type == 'rmsprop':
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            
            # 主迴圈
            for epoch in range(100):

                for j, data in enumerate(trainloader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    outputs = outputs.squeeze(dim=1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # 測試模型
                test_loss = 0.0
                with torch.no_grad():
                    for data in testloader:
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()
                
                avg_test_loss = test_loss / len(testloader)
                
                if avg_test_loss < best_cross_entropy_loss:
                    best_cross_entropy_loss = avg_test_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                if early_stopping_counter >= early_stopping:
                    break
            
            k_fold_cross_entropy_loss.append(best_cross_entropy_loss)
        
        return np.mean(k_fold_cross_entropy_loss)
    
    # 貝葉斯優化
    best = fmin(fn=objective, space=param_grid, algo=tpe.suggest, max_evals = max_evals)

    # 將索引轉換成實際儲存的內容
    best_params = {
    'total_hidden_layer': int(best['total_hidden_layer']),  # 四捨五入或取整
    'neurons_per_layer': int(best['neurons_per_layer']),    # 四捨五入或取整
    'dropout_rate': best['dropout_rate'],
    'learning_rate': best['learning_rate'],
    'optimizer': ['adam', 'sgd', 'rmsprop'][best['optimizer']]
    }

    return best_params

# 儲存最優超參數
def hyperparameter_save(best_params, save_path):
    """
    儲存最優超參數到一個JSON文件。
    
    參數:
        best_params (dict)  : 最優超參數的字典。
        file_path (str)     : 儲存文件的路徑。
    """
    with open(save_path, 'w') as f:
        json.dump(best_params, f, indent=4)

# 模型訓練
def model_train(train_loader, val_loader, best_params, input_dim, output_dim, early_stopping, max_epoch):
    
    """
    以最優超參數訓練一個模型。
    
    輸入:
    - train_data     : 訓練集。
    - val_data       : 驗證集。
    - best_params    : 最佳超參數組合。
    - input_dim      : 輸入維度。
    - output_dim     : 輸出維度。
    - early_stopping : 早停。
    
    輸出:
    - model          : 訓練好的模型
    - train_losses   : 每個epoch的訓練損失
    - val_losses     : 每個epoch的驗證損失。
    """

    #=========================================初始化=========================================#
    
    train_losses          =  []
    val_losses            =  []
    no_improvement_count  =  0
    best_val_loss         =  float('inf')  # 初始化最佳驗證損失為無窮大
    best_model_state      =  None  # 存儲最佳模型狀態

    #=========================================初始化=========================================#





    #=========================================主程式=========================================#

    # 提取超參數
    total_hidden_layers = best_params['total_hidden_layer']
    neurons_per_layer   = best_params['neurons_per_layer']
    dropout_rate        = best_params['dropout_rate']
    learning_rate       = best_params['learning_rate']
    optimizer_type      = best_params['optimizer']

    # GPU加速
    device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 創建模型
    model = MLP(input_dim, output_dim, total_hidden_layers, neurons_per_layer, dropout_rate).to(device)

    # 損失函數：多類別交叉熵
    criterion = nn.CrossEntropyLoss()

    # 優化器
    if   optimizer_type == 'adam':
        optimizer = optim.Adam      (model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD       (model.parameters(), lr=learning_rate)
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop   (model.parameters(), lr=learning_rate)

    # 訓練、驗證迴圈
    for epoch in range(max_epoch):
        
        #=========================================初始化=========================================#

        train_loss           = 0
        val_loss             = 0
        train_iteration_time = 0
        val_iteration_time   = 0

        #=========================================初始化=========================================#





        #=========================================主程式=========================================#

        # 切換至訓練模式
        model.train()

        # 訓練迴圈
        # 使用tqdm來包裝您的train_loader
        for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iteration_time += 1

        # 每個epoch後您可以打印出平均損失或其他指標
        print(f"Epoch {epoch+1}/{max_epoch}, Train Loss: {train_loss/train_iteration_time:.4f}")

        # 切換至驗證模式
        model.eval()

        # 驗證迴圈
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_iteration_time +=1

        # 計算該次epoch的平均驗證損失  
        val_losses.append(val_loss / val_iteration_time)
        
        # 早停機制
        if len(val_losses) > early_stopping:
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model_state = model.state_dict()
                no_improvement_count = 0  # 重置計數器
            elif val_losses[-1] >= min(val_losses[:-1]):
                no_improvement_count += 1

            if no_improvement_count >= early_stopping:
                model.load_state_dict(best_model_state)  # 加載最佳模型狀態
                break
        else:
            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                best_model_state = model.state_dict()
                no_improvement_count = 0  # 重置計數器

    return model, train_losses, val_losses

# 儲存Loss圖
def loss_plot_save(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Average Training and Validation Loss per Epoch')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# 模型測試
def model_test(model, test_loader):
    # 初始化
    y_true = []
    y_pred = []

    # 切換評估模式
    model.eval()

    # 使用GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 測試模型
    with torch.no_grad():  # 確保在評估過程中不計算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 獲取最可能的類別索引
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    return y_true, y_pred

# 儲存混淆矩陣
def confusion_matrix_save(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

# 計算、儲存指標
'''指標注釋

Kappa                                   : kappa值
Accuracy                                : 精確度
Sensitivity                             : 敏感度
Specificity                             : 特異度
Positive Predictive Value (PPV)         : 陽性預測準確率
Negative Predictive Value (NPV)         : 陰性預測準確率
Matthew's correlation coefficient (MCC) : 馬修斯相關係數
'''
def index_calculate_and_save(y_true, y_pred, save_path):
    # 計算混淆矩陣
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # 微平均和宏平均計算
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    precision_micro = precision_score(y_true, y_pred, average='micro')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    # 計算 MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    metrics = {
        'Accuracy': accuracy,
        'Kappa': kappa,
        'Macro Precision': precision_macro,
        'Macro Recall': recall_macro,
        'Macro F1': f1_macro,
        'Micro Precision': precision_micro,
        'Micro Recall': recall_micro,
        'Micro F1': f1_micro,
        'MCC': mcc
    }
    
    # 儲存指標到 JSON 文件
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)

#=========================================主函數=========================================#





#=========================================主程式=========================================#

def main():

    # 程式計時
    '''用於檢查程式運作時長'''
    main_start_time = time.time()

    #=========================================初始化=========================================#

    # 儲存所有模型的 y_true 和 y_pred
    all_y_true = []
    all_y_pred = []

    #=========================================初始化=========================================#





    #=========================================設定區=========================================#
    
    # 儲存路徑
    base_path    = "./訓練結果1"

    # 輸入特徵數
    input_dim    = 784

    # 輸出結果數
    output_dim   = 10

    # 訓練、驗證集比例
    train_size   = 0.8

    # 貝葉斯蒐索次數
    max_evals    = 5

    # 超參數範圍
    hidden_layers_choices     = (1, 5, 1)           # 在 1 到 5 之間隨機選取，步長為 1
    neurons_per_layer_choices = (100, 200, 10)      # 在 100 到 200 之間隨機選取，步長為 10
    dropout_rate_range        = (0.15, 0.45)             
    learning_rate_range       = (0.0005, 0.002)
    optimizer_choices         = ['adam', 'sgd', 'rmsprop']

    # 訓練模型數
    model_num    = 1

    # K折數
    K_FOLDS      = 5

    # 早停
    EARLY_STOPPING = 10

    # 單次訓練遍歷上限
    max_epoch    = 1

    #=========================================設定區=========================================#





    #=======================================程式運行區=======================================#
    
    # 資料預處理
    train_dataset, test_dataset = data_preprocess()
    print("已完成資料預處理")

    # 尋找最優超參數
    best_params = hyperparameter_best(train_dataset,
                                      input_dim,
                                      output_dim,
                                      max_evals, 
                                      hidden_layers_choices,
                                      neurons_per_layer_choices,
                                      dropout_rate_range,
                                      learning_rate_range,
                                      optimizer_choices,
                                      K_FOLDS,
                                      EARLY_STOPPING)
    print("已找到最佳超參數組合")

    # 儲存最優超參數
    save_path = generate_filename(base_path, stage = "find_hyper", model_num = None, content = "params", file_type="json")
    hyperparameter_save(best_params, save_path)

    # 主迴圈(訓練)
    for model_num in range(1, model_num + 1):

        # 程式計時
        '''用於檢查程式運作時長'''
        model_start_time = time.time()

        # 取得資料集標籤
        all_labels = [label for _, label in train_dataset]

        # 以分層採樣隨機分配訓練集和驗證集索引
        train_idx, val_idx = train_test_split(range(len(train_dataset)), test_size = 1 - train_size, stratify=all_labels)

        # 使用索引創建新的子集
        train_data  = Subset(train_dataset, train_idx)
        val_data    = Subset(train_dataset, val_idx)

        # 封裝訓練集、驗證集
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        val_loader   = DataLoader(val_data, batch_size=64, shuffle=False)

        # 模型訓練
        model, train_losses, val_losses = model_train(train_loader, val_loader, best_params, input_dim, output_dim, EARLY_STOPPING, max_epoch)

        # 儲存模型
        save_path = generate_filename(base_path, stage = "single_result", model_num = model_num, content = "model", file_type="pth")
        torch.save(model, save_path)

        # 儲存Loss圖
        save_path = generate_filename(base_path, stage = "single_result", model_num = model_num, content = "Loss", file_type="png")
        loss_plot_save(train_losses, val_losses, save_path)

        # 測試模型
        test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)
        y_true, y_pred = model_test(model, test_loader)

        # 將 y_true 和 y_pred 儲存到總列表中
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        # 儲存混淆矩陣
        save_path = generate_filename(base_path, stage = "single_result", model_num = model_num, content = "confusion_matrix", file_type="png")
        confusion_matrix_save(y_true, y_pred, save_path)

        # 計算、儲存各式指標
        save_path = generate_filename(base_path, stage = "single_result", model_num = model_num, content = "metrics", file_type="json")
        index_calculate_and_save(y_true, y_pred, save_path)

        # 程式計時
        '''用於檢查程式運作時長'''
        model_end_time = time.time()

        # 計算並顯示總花費時長
        model_elapsed_time = model_end_time - model_start_time
        print(f"訓練第{model_num}個模型花費 {model_elapsed_time} 秒")
    
    print("主迴圈運行完畢")

    # 儲存混淆矩陣
    save_path = generate_filename(base_path, stage = "total_result", model_num = None, content = "total_confusion_matrix", file_type="png")
    confusion_matrix_save(all_y_true, all_y_pred, save_path)

    # 計算、儲存各式指標
    save_path = generate_filename(base_path, stage = "total_result", model_num = None, content = "total_metrics", file_type="json")
    index_calculate_and_save(all_y_true, all_y_pred, save_path)

    #=======================================程式運行區=======================================#

    # 程式計時
    '''用於檢查程式運作時長'''
    main_end_time = time.time()

    # 計算並顯示總花費時長
    main_elapsed_time = main_end_time - main_start_time
    print(f"總共花費 {main_elapsed_time} 秒")

#=========================================主程式=========================================#

if __name__ == "__main__":
    main()
