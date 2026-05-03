"""
    历史记录清空器
"""
import os
import shutil


class Cleaner:
    def __init__(self):
        self.root = './pool'
        self.dir_lst = [
            'meta/tf', 'meta/torch',
            'sarsd',
            'weights/tf', 'weights/torch'
        ]
        self.check_lst = [
            'meta',
            'meta/tf', 'meta/torch',
            'sarsd',
            'weights',
            'weights/tf', 'weights/torch'
        ]

    def clean(self, root):
        for root, dirs, files in os.walk(root, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            for name in dirs:
                dir_path = os.path.join(root, name)
                os.rmdir(dir_path)

    def just_give_me_a_call(self):
        for path in self.dir_lst:
            self.clean(os.path.join(self.root, path))
        self.create()
        # self.dont_look_back_in_anger()
        print('I have cleaned up and added new things!! ^_^')

    def create(self):
        for path in self.check_lst:
            p = os.path.join(self.root, path)
            if not os.path.exists(p):
                os.mkdir(p)

    def dont_look_back_in_anger(self):
        src_file = './history_backup'
        dst_file = os.path.join(self.root, 'sarsd', 'history')
        shutil.copyfile(src_file, dst_file)


if __name__ == '__main__':
    cleaner = Cleaner()
    cleaner.just_give_me_a_call()
