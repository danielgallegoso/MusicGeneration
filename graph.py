import matplotlib.pyplot as plt
import numpy as np

def main():
    train = [
        0.00312803535721,
        0.000527809618239,
        9.30916430301e-05,
        0.000461019585299,
        5.43087998537e-05,
        0.000102149688394,
        4.69659802e-05,
        6.5532006609e-05,
        5.06822561629e-05,
        4.17010801232e-05,
        4.10297473519e-05,
        3.24023784583e-05,
        3.38186256741e-05,
        3.99826266779e-05,
        5.00306105898e-05,
        1.63666692965e-05,
        2.45338703552e-05,
        1.90572607012e-05,
        2.93455373786e-05,
        1.36183672709e-05,
        8.57351744763e-06,
        1.17195233553e-05,
        1.73536374263e-05,
        1.57378402523e-05,
        1.02325323331e-05,
        3.81288295968e-05
    ]

    test = [
        0.00295887848256,
        0.000563351275005,
        0.000243286959014,
        0.0021715302419,
        0.000126920326452,
        0.000137492437921,
        0.000158025350545,
        0.000117627508137,
        0.000155826979526,
        0.000172643471296,
        0.000123912241852,
        0.000196716028516,
        0.000210531727527,
        0.000331476914864,
        0.000510500742387,
        0.000157927872181,
        0.000119387195603,
        9.43515782229e-05,
        0.000267943324303,
        0.000278322103633,
        0.000345144098669,
        0.000177899626097,
        0.000198881201134,
        0.000195006254056,
        0.000186382904159,
        0.000579208246965
    ]
    epoch = np.arange(26)
    plt.plot(epoch, train, label='Dev')
    plt.plot(epoch, test, label='Test')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__== "__main__":
  main()
