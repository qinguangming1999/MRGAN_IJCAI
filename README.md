
# Read_me

This is a pytorch implementation of submission: <b>Multi-Relational Graph Attention Network for Social Relationship Inference from Human Mobility Data</b>


<i>IJCAI'24</i>

<div align=center><img src="https://github.com/qinguangming1999/MRGAN_IJCAI/blob/main/overview.png" width="900"/></div>

## abstract
Inferring social relationships from human mobility data holds significant value in real-life spatio-temporal applications, inspiring the development of a series of graph-based methods for deriving such relationships.
However, despite their noted effectiveness, we argue that previous methods either rely solely on direct relations between users, neglecting valuable user mobility patterns, or have not fully harnessed the indirect interactions, thereby struggling to capture users' mobility preferences. To address these issues, in this work, we propose the Multi-Relational Graph Attention Network MRGAN, a novel graph attention network, which is able to explicitly model indirect relations and effectively capture their different impact. Specifically, we first extract a multi-relational graph from heterogeneous mobility graph to explicitly model the direct and indirect relations, %as different mobility patterns
and then utilize influence attention and cross-relation attention to further capture the different influence between users, and different importance of relations for each user. 
Comprehensive experiments on three real-world mobile datasets demonstrate that the proposed model significantly outperforms state-of-the-art models in predicting social relationships between users.



## Requirements
  * Python 3.8.6
  * torch 2.1.2
  * pandas 2.3.1
  * keras 2.11.0
  * networkx
  * numpy
  * scipy
  * scikit-learn
  * dgl 1.1.2
 
## Run the code
```bash
Python train_MRGAN.py to run the code
```

## Contact

Please email qinguangming0413@163.com or qinguangming@stu.ouc.edu.cn if you have any questions

## Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)




```BibTeX

@inproceedings{qin2024multi,
  title={Multi-relational graph attention network for social relationship inference from human mobility data},
  author={Qin, Guangming and Qi, Jianpeng and Wang, Bin and Jiang, Guiyuan and Yu, Yanwei and Dong, Junyu},
  booktitle={Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, Kate Larson (Ed.). International Joint Conferences on Artificial Intelligence Organization},
  pages={2315--2323},
  year={2024}
}
```

<br>




