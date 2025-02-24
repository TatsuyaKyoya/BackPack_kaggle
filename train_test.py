# モデルの訓練及びテストを行うプログラム

import pickle as pkl
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import xgboost as xgb
from pathlib import Path
import re
from datetime import datetime
import time

DIR = Path(__file__).resolve().parent / 'dataset/processed_dataset'
SUBMISSION = Path(__file__).resolve().parent / 'submissions'
MODEL = Path(__file__).resolve().parent / 'models'

def load_data(num='latest'):
    flist = [p.name for p in DIR.iterdir() if p.is_file()]
    if num == 'latest':
        filename = flist[-1]
    else:
        assert isinstance(num, int), 'num must be int.'
        for f in flist:
            if num in f:
                filename = f
    with open(DIR / filename, 'rb') as f:
        data = pkl.load(f)
    return data

def compute_rmse(oof, answer):
    rmse = np.sqrt(np.mean(np.square(oof - answer)))
    return rmse

def save(object: pd.DataFrame, obj_type, **kwargs):
    today = datetime.today().strftime('%Y%m%d')
    if obj_type == 'submit':
        dir = SUBMISSION
        title = f'submission_RMSE{kwargs["rmse"]}'
        suffix = 'csv'
    if not dir.exists():
        dir.mkdir(parents=True)
    flist = [p.name for p in dir.iterdir() if p.is_file()]
    if len(flist) == 0:
        filename = f'{title}_{today}_0.{suffix}'
    else:
        latest = int(re.findall(r'.*(\d*).*', flist[-1])) + 1
        filename = f'{title}_{today}_{latest}.{suffix}'
    object.to_csv(dir/filename, encoding='utf-8', index=False)
    print('done')

def main():
    start = time.time()
    data = load_data()
    oof = np.zeros(data['train_num'])
    answer = np.zeros(data['train_num'])
    pred = np.zeros(data['test_num'])
    COLS = data['train_cols']
    folds = [k for k in data.keys() if re.search(r'.*[0-9]+.*', k) != None]

    for f in folds:
        print(f'### FOLDS: {f} ###')
        train = {'X': pd.DataFrame(data[f]['train']['X'], columns=COLS), 'y': pd.DataFrame(data[f]['train']['y'])}
        valid = {'X': pd.DataFrame(data[f]['valid']['X'], columns=COLS), 'y': pd.DataFrame(data[f]['valid']['y'])}
        test_col = ['id']
        test_col.extend(COLS)
        test = pd.DataFrame(data[f]['test'], columns=test_col)
        id = [int(i) for i in test['id'].values.tolist()]
        model = XGBRegressor(
            device="cpu",
            max_depth=2, # 6
            colsample_bytree=0.5, 
            subsample=0.8,  
            n_estimators=100, # 10_000  
            learning_rate=0.02,  
            enable_categorical=True,
            min_child_weight=10,
            early_stopping_rounds=100,
        )
        model.fit(train['X'][COLS], train['y'],
                      eval_set=[(valid['X'][COLS], valid['y'])],
                      verbose=300)
        oof[data[f]['test_index']] = model.predict(valid['X'][COLS])
        answer[data[f]['test_index']] = valid['y'].values.squeeze()
        pred += model.predict(test[COLS])
    pred /= len(folds)
    rmse = str(round(compute_rmse(oof=oof, answer=answer), 1)).replace('.', '-')
    submmit = pd.DataFrame({'id': id, 'Price': pred})
    save(submmit, 'submit', rmse=rmse)
    end = time.time()
    print(f'exec time: {end - start}')

if __name__ == '__main__':
    main()