import csv
import os


def split_csv(path='data/ratings.csv', total_len=100836, per=0.8):
    #此方法未打乱数据顺序
    # 如果train.csv和vali.csv存在就删除
    if os.path.exists('data/train.csv'):
        os.remove('data/rain.csv')
    if os.path.exists('data/vali.csv'):
        os.remove('data/vali.csv')

    with open(path, 'r', newline='') as file:
        csvreader = csv.reader(file)
        i = 0
        for row in csvreader:
            if i < round(total_len * per/100):
                # train.csv存放路径
                csv_path = os.path.join("data", 'train.csv')
                print(csv_path)
                # 不存在此文件的时候，就创建
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
                # 存在的时候就往里面添加
                else:
                    with open(csv_path, 'a', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
            elif (i >= round(total_len * per/100)) and (i < total_len):
            	# vali.csv存放路径
                csv_path = os.path.join("data", 'vali.csv')
                print(csv_path)
                # 不存在此文件的时候，就创建
                if not os.path.exists(csv_path):
                    with open(csv_path, 'w', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
                # 存在的时候就往里面添加
                else:
                    with open(csv_path, 'a', newline='') as file:
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                    i += 1
            else:
                break

    print("训练集和验证集分离成功")
    return

if __name__ == '__main__':

    path = 'data/ratings.csv'
    total_len = len(open(path, 'r').readlines())# csv文件行数
    per = 80 # 分割比例%
    split_csv(path, total_len, per)
