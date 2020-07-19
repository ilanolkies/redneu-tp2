from hebbiano import Hebbiano

hebbiano = Hebbiano('./dataset/tp2_training_dataset.csv')

# alg, M, lr, min_ort, max_epoch, trace (= 0)
#hebbiano.train('oja', 9, 0.00001, 0.001, 600, 1)
#hebbiano.plot()

hebbiano.train('sanger', 9, 0.001, 0.001, 600, 1)
hebbiano.plot()
