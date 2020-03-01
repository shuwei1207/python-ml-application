# python-ml-application

本次是用learning to rank（L2R）的方式來訓練資料，來當作一種選股策略，L2R有許多方法和演算法，我藉由python的套件Xgboost來加以實現。


一、	資料集：
機密資料。
 

二、	程式說明：
Step1：封包處理
先引入我們需要的封包
 
 
Step2：資料處理:


Step3：資料分割（X & y）


Step4：放入資料（有兩種方法）
法一：
直接使用DMatrix來分析資料。
 
法二：
		使用svm檔案儲存資料，後分析資料。
 
 
Step5：設置參數
		之後便可以設置參數，一般是調整eta也就是learning rate來提高準確率，並可搭配不同的函數，也就是objective，且我們是multiclass的分類，必須填入class的數目。


Step6：開始訓練


Step7：Confusion Matrix
		用此矩陣可以幫助我們了解best_preds的label和y_test的label是否有對應，因為有5個class所以此矩陣會有25個，對角線（＼）代表預測精準的個數，可以幫助我們了解訓練出來的結果。
 
 
三、	結果呈現（可調整的參數）：
指標一：精準度
		
 
指標二：Confusion Matrix


指標三：挑出個股
（可能的error：因沒有設計防呆功能，故label０的個股未滿3股時可能成是無法執行）
