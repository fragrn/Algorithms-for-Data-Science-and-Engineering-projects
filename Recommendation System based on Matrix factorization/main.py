import pandas as pd
import numpy as np
# from tqdm import tqdm
import argparse   #命令行参数模块

from model import MatrixDecomForRecSys
from metrics import RMSE
import time

if __name__ == '__main__':

    # set hyper-parameter
    parser = argparse.ArgumentParser(description="Command")
    parser.add_argument('--learning_rate', default=0.02, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--reg_p', default=0.01, type=float)
    parser.add_argument('--reg_q', default=0.01, type=float)
    parser.add_argument('--hidden_size', default=16, type=int)
    parser.add_argument('--optimizer_type', default="SGD", type=str, help="SGD or BGD")
    parser.add_argument('--train', default=False, action='store_true', help='is train') # default=False，在没有指定参数时，默认什么也不做
    parser.add_argument('--test', default=False, action='store_true', help='is test')
    parser.add_argument('--recommend',default=False,type=int) #仅为test里面出现的user进行推荐

    args = parser.parse_args()

    # reading training data
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
    print("Reading training data ...")
    train_dataset = pd.read_csv("data/train.csv", usecols=range(3), dtype=dict(dtype))
    print("Reading developing data ...")
    dev_dataset = pd.read_csv("data/dev.csv", usecols=range(3), dtype=dict(dtype))

    model = MatrixDecomForRecSys(
            lr=args.learning_rate, 
            batch_size=args.batch_size,
            reg_p=args.reg_p, 
            reg_q=args.reg_q, 
            hidden_size=args.hidden_size,
            epoch=args.epoch,
            columns=["userId", "movieId", "rating"],
            metric=RMSE   #此处的metric指定为RMSE函数
            )
    model.load_dataset(train_data=train_dataset, dev_data=dev_dataset)


    if args.train:
        time_start = time.time() #开始计时
        print("Starting training ...")
        model.train(optimizer_type=args.optimizer_type)
        print("Finish training.")
        time_end = time.time()    #结束计时
        time_c= time_end - time_start   #运行所花时间
        print('time cost', time_c, 's')
    
    if args.test:
        dtype = [("userId", np.int32), ("movieId", np.int32)]
        print("Reading testing data ...")
        test_dataset = pd.read_csv("data/test.csv", usecols=range(3), dtype=dict(dtype))

        print("Starting predicting ...")
        model.test(test_dataset)
        print("Finish predicting, you can submit your results on the leaderboard.")

    if args.recommend:
        print("start recommending")
        model.recommend(args.recommend)
        print("finish recommending")