for LRate in 0.00001
do
	for Epoch in 500 700 900
	do
		for DropoutRate in 0.1 0.2
		do
			for n_hidden2 in  150 200
			do
				for n_hidden3 in 150 200
				do
					for n_hidden4 in 150 200
					do
						python "/Users/scml/Downloads/KDD/dnn_3.py"  $LRate $Epoch $DropoutRate $n_hidden2 $n_hidden3 $n_hidden4
					done
				done
			done
		done
	done 
done