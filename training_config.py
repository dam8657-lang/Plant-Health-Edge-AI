#model.train(data='./dataset/yymnist/data.yaml',epochs=50,imgsz=256,patience=10,batch=16)
train_config={
    'batch':16, #주로변경    
    'epochs':100, #주로변경   
    'imgsz':224,
    #'epochs':600, #주로변경   
    'cos_lr':True,    
    'cache':True,
    'freeze':None,
    'plots':True,
    'close_mosaic':10, #주로변경   
    'resume':False,
    'patience':20, #주로변경
    'pretrained':True,
    'lr0':0.01,
    'label_smoothing':0.05,
    'project':'DOG1367',    
}