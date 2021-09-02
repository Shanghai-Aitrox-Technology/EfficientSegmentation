import lmdb
import json

from Common.file_utils import MyEncoder


class DataBaseUtils(object):
    def __init__(self):
        super(DataBaseUtils, self).__init__()

    @staticmethod
    def creat_db(db_dir):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        txn.commit()
        env.close()

    @staticmethod
    def update_record_in_db(db_dir, idx, data_dict):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        txn.put(str(idx).encode(), value=json.dumps(data_dict, cls=MyEncoder).encode())
        txn.commit()
        env.close()

    @staticmethod
    def delete_record_in_db(db_dir, idx):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        txn.delete(str(idx).encode())
        txn.commit()
        env.close()

    @staticmethod
    def get_record_in_db(db_dir, idx):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin()
        value = txn.get(str(idx).encode())
        env.close()
        if value is None:
            return None
        value = str(value, encoding='utf-8')
        data_dict = json.loads(value)

        return data_dict

    @staticmethod
    def read_records_in_db(db_dir):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin()
        out_records = dict()
        for key, value in txn.cursor():
            key = str(key, encoding='utf-8')
            value = str(value, encoding='utf-8')

            label_info = json.loads(value)
            out_records[key] = label_info
        env.close()

        return out_records

    @staticmethod
    def write_records_in_db(db_dir, in_records):
        env = lmdb.open(db_dir, map_size=int(1e9))
        txn = env.begin(write=True)
        for key, value in in_records.items():
            txn.put(str(key).encode(), value=json.dumps(value, cls=MyEncoder).encode())
        txn.commit()
        env.close()
