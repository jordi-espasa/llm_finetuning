nvidia 3060

Using dataset wds_vtab-cifar100
With 100 classes and 18 templates

python run_evaluate.py --dataset wds_vtab-cifar100 --ensemble_method baseline
Setup time: 0.11305618286132812
[08:35,  1.67it/s]
{'accuracy': 0.5396}

python run_evaluate.py --dataset wds_vtab-cifar100 --ensemble_method mean_input
Setup time: 1.4808909893035889
[08:31,  1.68it/s]        
{'accuracy': 0.6756727272727273}

python run_evaluate.py --dataset wds_vtab-cifar100 --ensemble_method mean_logit
Setup time: 1.4081366062164307
[08:37,  1.66it/s]
{'accuracy': 0.6765454545454546}

python run_evaluate.py --dataset wds_vtab-cifar100 --ensemble_method mean_softmax
Setup time: 1.4496018886566162
[08:27,  1.70it/s]
{'accuracy': 0.6772545454545454}


Using dataset wds_vtab-caltech101
With 102 classes and 34 templates

python run_evaluate.py --dataset wds_vtab-caltech101 --ensemble_method baseline
Setup time: 0.09844589233398438
[01:48,  1.28it/s]
{'accuracy': 0.8092328581126952}

python run_evaluate.py --dataset wds_vtab-caltech101 --ensemble_method mean_input
Setup time: 1.612267255783081
[01:46,  1.31it/s]
{'accuracy': 0.8475899524779362}

python run_evaluate.py --dataset wds_vtab-caltech101 --ensemble_method mean_logit
Setup time: 1.4479436874389648
[01:45,  1.32it/s]
{'accuracy': 0.8449875537451912}

python run_evaluate.py --dataset wds_vtab-caltech101 --ensemble_method mean_softmax
Setup time: 1.530440330505371
[01:43,  1.34it/s]
{'accuracy': 0.8477031002489251}

