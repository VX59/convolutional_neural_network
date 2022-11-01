from Sorters import *

class Sorter(Sorter_Framework):
    def __init__(self,neural_model,input_size=0,dataset_start=0,name='',groups=0,class_num=0,dimension='0x0',gcm=False,lr=1.3e-3):
        super().__init__(name=name,input_size=input_size,dimension=dimension,class_num=class_num)
        self.dataset_start = dataset_start
        self.groups = groups
        self.gcm = gcm
        self.load_neural_model(neural_model,lr=lr)
    
    def train(self,persistance,k=0,kfold=True):
        if kfold: print('kfold mode on')
        return self.train_model(persistance,kfold=kfold,k=k)

    def unpack_folders(self,container,target_out):
        folders = os.listdir(container)
        for folder in folders:
            contents = os.listdir(container+folder)
            for img in contents:
                path = container+folder+"/"+img
                print(path)
                shutil.move(path,target_out+img)

    def preprocess(self,filters,working_dir='dataset/'):
        self.preprocess_data(self.target_train,working_dir,start=self.dataset_start,groups=self.groups,filters=filters,alpha_split=2.0)
        self.make_labels(self.target_train, groups=self.groups,feeder_mode=self.gcm)
        self.preprocess_data(self.target_test,working_dir,offset=1,start=self.dataset_start,groups=self.groups,filters=filters,alpha_split=2.0)
        self.make_labels(self.target_test, groups=self.groups,offset=1,feeder_mode=self.gcm)
        
    def load_preprocessed_data(self):
        self.test_logs()
        self.train_logs()
        self.load_data()

    def synthesize(self,step):
        self.synthesize_data(self.target_train_synth,step)
        self.make_labels(self.target_train_synth, groups=self.groups,feeder_mode=self.gcm)
        self.train_synth_logs()
        self.load_synth_data()
    
    def evaluate(self,test_data,test_labels):
        self.make_prediction(test_data,test_labels)

class General_Sorting(Sorter):
    def __init__(self,input_size,dataset_start):
        self.G4X2 = Sorter(linear_model,input_size=input_size,dataset_start=dataset_start,name='4X2 sorter',groups=4,class_num=2,dimension='4X2',gcm=True)
        self.G6X3 = Sorter(deep_model_3,input_size=input_size,dataset_start=dataset_start,name='6X3 sorter',groups=6,class_num=3,dimension='6X3',gcm=True)
        self.G8X4 = Sorter(deep_model_4,input_size=input_size,dataset_start=dataset_start,name='8X4 sorter',groups=8,class_num=4,dimension='8X4',gcm=True)
        self.G10X5 = Sorter(deep_model_5,input_size=input_size,dataset_start=dataset_start,name='10X5 sorter',groups=10,class_num=5,dimension='10X5',gcm=True)
        self.G12X6 = Sorter(deep_model_6,input_size=input_size,dataset_start=dataset_start,name='12X6 sorter',groups=12,class_num=6,dimension='12X6',gcm=True)
        self.G16X8 = Sorter(deep_model_8,input_size=input_size,dataset_start=dataset_start,name='16X8 sorter',groups=16,class_num=8,dimension='16X8',gcm=True)
        self.G24X6 = Sorter(deep_model_6,input_size=input_size,dataset_start=dataset_start,name='24X6 sorter',groups=24,class_num=6,dimension='24X6',gcm=True)
        self.I2X2 = Sorter(linear_model,input_size=input_size,dataset_start=dataset_start,name='2X2 sorter',groups=2,class_num=2,dimension='2X2')
        self.I3X3 = Sorter(deep_model_3,input_size=input_size,dataset_start=dataset_start,name='3X3 sorter',groups=3,class_num=3,dimension='3X3')
        self.I4X4 = Sorter(deep_model_4,input_size=input_size,dataset_start=dataset_start,name='4X4 sorter',groups=4,class_num=4,dimension='4X4')
        self.I5X5 = Sorter(deep_model_5,input_size=input_size,dataset_start=dataset_start,name='5X5 sorter',groups=5,class_num=5,dimension='5X5')
        self.I6X6 = Sorter(deep_model_6,input_size=input_size,dataset_start=dataset_start,name='6X6 sorter',groups=6,class_num=6,dimension='6X6')
        self.I8X8 = Sorter(deep_model_8,input_size=input_size,dataset_start=dataset_start,name='8X8 sorter',groups=8,class_num=8,dimension='8X8')