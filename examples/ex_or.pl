=begin
    
    File:	examples/ex_and.pl
	Author: Josiah Bryan, <jdb@wcoil.com>
	Desc: 

		This demonstrates a simple OR gate.

=cut

	use AI::neuralNet::Mesh;
	
	# Uses 1 layer and 2 nodes per layer, with one output node
	my $net = new AI::NeuralNet::Mesh(2,2,1);
	
	if(!$net->load('or.mesh')) {
		$net->learn_set([	
			[1,1], [1],
			[1,0], [1],
			[0,1], [1],
			[0,0], [0],
		]);
		$net->save('or.mesh');
	}

	print "Learning complete.\n";
	print "Testing with a gate value of (0,0):",$net->run([0,0])->[0],"\n";
	print "Testing with a gate value of (0,1):",$net->run([0,1])->[0],"\n";
	print "Testing with a gate value of (1,0):",$net->run([1,0])->[0],"\n";
	print "Testing with a gate value of (1,1):",$net->run([1,1])->[0],"\n";
	
	