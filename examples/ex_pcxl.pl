=begin
	
	File:   examples/ex_pcxl.pl
	Author: Josiah Bryan, jdb@wcoil.com
	Desc:	
		
		This just demonstrates simple usage of the pcx loader.
	
=cut

	use AI::NeuralNet::Mesh;
	
	my $net=AI::NeuralNet::Mesh->new(2,2);
	
	my $img = $net->load_pcx("josiah.pcx");             
	
	$net->join_cols($img->get_block([0,0,50,50]),50,0);
	
