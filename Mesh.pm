#!/usr/bin/perl	

# Copyright (c) 2000  Josiah Bryan  USA
#
# See AUTHOR section in pod text below for usage and distribution rights.   
#

BEGIN {
	 $AI::NeuralNet::Mesh::VERSION = "0.31";
	 $AI::NeuralNet::Mesh::ID = 
'$Id: AI::NeuralNet::Mesh.pm, v'.$AI::NeuralNet::Mesh::VERSION.' 2000/25/12 05:26:10 josiah Exp $';
}

package AI::NeuralNet::Mesh;
    
    use strict;
    use Benchmark;
    
	# Debugging subs
	$AI::NeuralNet::Mesh::DEBUG  = 0;
	sub whowasi { (caller(1))[3] . '()' }
	sub debug { shift; $AI::NeuralNet::Mesh::DEBUG = shift || 0; } 
	sub d { shift if(substr($_[0],0,4) eq 'AI::'); my ($a,$b,$c)=(shift,shift,$AI::NeuralNet::Mesh::DEBUG); print $a if($c == $b); return $c }
	
	# Return version of ::ID string passed or current version of this
	# module if no string is passed.
	sub version {
		shift if(substr($_[0],0,4) eq 'AI::');
		return substr((split(/\s/,(shift || $AI::NeuralNet::Mesh::ID)))[2],1);
	}
	
	# Rounds a floating-point to an integer with int() and sprintf()
	sub intr  {
    	shift if(substr($_[0],0,4) eq 'AI::');
      	try   { return int(sprintf("%.0f",shift)) }
      	catch { return 0 }
	}
    
	# Package constructor
	sub new {
		no strict 'refs';
		my $type	=	shift;
		my $self	=	{};
		my $layers	=	shift;
		my $nodes	=	shift;
		my $outputs	=	shift || $nodes;
		my $inputs	=	shift || $nodes;
        
		bless $self, $type;
		                       
		# If $layers is a string, then it will be numerically equal to 0, so 
		# try to load it as a network file.
		if($layers == 0) {  
		    # We use a "1" flag as the second argument to indicate that we 
		    # want load() to call the new constructor to make a network the
		    # same size as in the file and return a refrence to the network,
		    # instead of just creating the network from pre-exisiting refrence
			return $self->load($layers,1);
		}
		
		# Save parameters
		$self->{total_nodes}	= $layers * $nodes + $outputs;
		$self->{total_layers}	= $layers;
		$self->{nodes}			= $nodes;
		$self->{inputs}			= $inputs;
		$self->{outputs}		= $outputs;
		
		# Looks like we got ourselves a layer specs array
		if(ref($layers) eq "ARRAY") { 
			if(ref($layers->[0]) eq "HASH") {
				$self->{total_nodes}	=	0;
				$self->{inputs}			=	$layers->[0]->{nodes};
				$self->{nodes}			=	$layers->[0]->{nodes};
				$self->{outputs}		=	$layers->[$#{$layers}]->{nodes};
				$self->{total_layers}	=	$#{$layers};
				for (0..$#{$layers}){$self->{layers}->[$_] = $layers->[$_]->{nodes}}
				for (0..$self->{total_layers}){$self->{total_nodes}+=$self->{layers}->[$_]}	
			} else {
				$self->{inputs}			= $layers->[0];
			    $self->{nodes}			= $layers->[0];
				$self->{outputs}		= $layers->[$#{$layers}];
				$self->{layers} 		= $layers;
				$self->{total_layers}	= $#{$self->{layers}};
				$self->{total_nodes}	= 0;
				for (0..$self->{total_layers}) {
					$self->{total_nodes}+=$self->{layers}->[$_];
				}
			}
		}
		
		# Initalize misc. variables
		$self->{col_width}		=	5;
		$self->{random}			=	0;
		$self->{const}			=	0.0001;
		
		# Build mesh
		$self->_init();	
		
		# Initalize activation, thresholds, etc, if provided
		if(ref($layers->[0]) eq "HASH") {
			for (0..$#{$layers}) {
				$self->activation($_,$layers->[$_]->{activation});
				$self->threshold($_,$layers->[$_]->{threshold});
				$self->mean($_,$layers->[$_]->{mean});
			}
		}
				
		# Done!
		return $self;
	}	
    

    # Internal usage
    # Connects one range of nodes to another range
    sub _c {
    	my $self	=	shift;
    	my $r1a		=	shift;
    	my $r1b		=	shift;
    	my $r2a		=	shift;
    	my $r2b		=	shift;
    	my $m1		=	shift || $self->{mesh};
    	my $m2		=	shift || $m1;
    	
    	if(($r2b-$r2a) >= ($r1b-$r1a)) {
    		d("Case 1.($r1a..$r1b),($r2a..$r2b)..\n",10);
	    	for my $y ($r1a..$r1b-1) {
				for my $z ($r2a..$r2b-1) {
					d(".($y,$z).\n",10);
					$m1->[$y]->add_output_node($m2->[$z]);
				}
			}
		}
		elsif(($r2b-$r2a)<($r1b-$r1a)) {
			d("Case 2.($r1a..$r1b),($r2a..$r2b)..\n",10);
			my $div = intr(($r1b-$r1a)/($r2b-$r2a));
			for my $x (0..($r2b-$r2a)-1) {
				for my $y (0..($r1b-$r1a)-1) {
					d(".($x,$y).$r2a+($x*$div+$y)\n",10);
					$m1->[$r1a+($x*$div+$y)]->add_output_node($m2->[$r2a+$x]) if($m1->[$r1a+($x*$div+$y)]);
			 	}
			} 
		} else {
		    $self->{error} = "_c(): Range defenition error. ($r1a..$r1b),($r2a..$r2b).";
		    return undef;
		}		
    }
    
    # Internal usage
    # Creates the mesh of neurons
    sub _init {
    	my $self		=	shift;
    	my $nodes		=	$self->{nodes};
    	my $outputs		=	$self->{outputs} || $nodes;
    	my $inputs		=	$self->{inputs}  || $nodes;
    	my $layers		=	$self->{total_layers};
        my $tmp 		=	$self->{total_nodes} || ($layers * $nodes + $outputs);
    	my $layer_specs	=	$self->{layers};
        my ($x,$y,$z);
        no strict 'refs';
        
        # Just to be safe.
        $self->{total_nodes} = $tmp;
        
        # If they didn't give layer specifications, then we derive our own specs.
        if(!(defined $self->{layers})) {
        	$layer_specs = [split(',',"$nodes," x $layers)];
        	$layer_specs->[$#{$layer_specs}+1]=$outputs;
        	$self->{layers}	= $layer_specs;
        }
            
		# First create the individual nodes
		for my $x (0..$tmp-1) {
			$self->{mesh}->[$x] = AI::NeuralNet::Mesh::node->new($self);
        }              
        
        # Get an instance of an output (data collector) node
		$self->{output} = AI::NeuralNet::Mesh::output->new($self);
		
		# Connect the output layer to the data collector
        for $x (0..$outputs-1) {                    
			$self->{mesh}->[$tmp-$outputs+$x]->add_output_node($self->{output});
		}
		
		# Now we use the _c() method to connect the layers together.
        $y=0;
        for $x (0..$layers-1) {
        	$z = $layer_specs->[$x];
        	$self->_c($y,$y+$z,$y+$z,$y+$z+$layer_specs->[$x+1]);
        	$y+=$z;
		}		
		
		# Get an instance of our cap node.
		$self->{input}->{cap} = AI::NeuralNet::Mesh::cap->new(); 

		# Add a cap to the bottom of the mesh to stop it from trying
		# to recursivly adjust_weight() where there are no more nodes.		
		for my $x (0..$inputs-1) {
			$self->{input}->{IDs}->[$x] = 
				$self->{mesh}->[$x]->add_input_node($self->{input}->{cap});
		}
	}
    
    # See POD for usage
    sub extend {
    	my $self	=	shift;
    	my $layers	=	shift;
    
    	# Looks like we got ourselves a layer specs array
		if(ref($layers) eq "ARRAY") { 
			if($self->{total_layers}!=$#{$layers}) {
				$self->{error} = "extend(): Cannot add new layers. Create a new network to add layers.\n";
				return undef;
			}
			if(ref($layers->[0]) eq "HASH") {
				$self->{total_nodes}	=	0;
				$self->{inputs}			=	$layers->[0]->{nodes};
				$self->{nodes}			=	$layers->[0]->{nodes};
				$self->{outputs}		=	$layers->[$#{$layers}]->{nodes};
				for (0..$#{$layers}){
					$self->extend_layer($_,$layers->[$_]);
					$self->{layers}->[$_] =$layers->[$_]->{nodes};
				}
				for (0..$self->{total_layers}){$self->{total_nodes}+=$self->{layers}->[$_]}	
			} else {
				$self->{inputs}			= $layers->[0];
			    $self->{nodes}			= $layers->[0];
				$self->{outputs}		= $layers->[$#{$layers}];
				$self->{total_nodes}	= 0;
				for (0..$self->{total_layers}){$self->extend_layer($_,$layers->[$_])}
				$self->{layers} 		= $layers;
				for (0..$self->{total_layers}){$self->{total_nodes}+= $self->{layers}->[$_]}
			}
		} else {
			$self->{error} = "extend(): Invalid argument type.\n";
			return undef;
		}
		return 1;
	}
    
    # See POD for usage
    sub extend_layer {
    	my $self	=	shift;
    	my $layer	=	shift || 0;
    	my $specs	=	shift;
    	if(!$specs) {
    		$self->{error} = "extend_layer(): You must provide specs to extend layer $layer with.\n";
    		return undef;
    	}
    	if(ref($specs) eq "HASH") {
    		$self->activation($layer,$specs->{activation}) if($specs->{activation});
    		$self->threshold($layer,$specs->{threshold})   if($specs->{threshold});
    		$self->mean($layer,$specs->{mean})             if($specs->{mean});
    		return $self->add_nodes($layer,$specs->{nodes});
    	} else { 
    		return $self->add_nodes($layer,$specs);
    	}
    	return 1;
    }
    
    # Pseudo-internal usage
    sub add_nodes {
    	my $self	=	shift;
    	my $layer	=	shift;
    	my $nodes	=	shift;
    	d("Checking on extending layer $layer to $nodes nodes (check:$self->{layers}->[$layer]).\n",9);
        return 1 if ($nodes == $self->{layers}->[$layer]);
        if ($self->{layers}->[$layer]>$nodes) {
        	$self->{error} = "add_nodes(): I cannot remove nodes from the network with this version of my module. You must create a new network to remove nodes.\n";
        	return undef;
        }
        my $more	=	$nodes - $self->{layers}->[$layer] - 1;
        for (0..$more) {
        	$self->{mesh}->[$#{$self->{mesh}}+1] = AI::NeuralNet::Mesh::node->new($self);
        }
        d("Extending layer $layer by $more.\n",9);
        my $n		=	0;
		no strict 'refs';
		for(0..$layer-2){$n+=$self->{layers}->[$_]}
		$self->_c($n,$n+$self->{layers}->[$layer-1],$#{$self->{mesh}}-$more+1,$#{$self->{mesh}});
		$self->_c($#{$self->{mesh}}-$more+1,$#{$self->{mesh}},$n+$self->{layers}->[$layer],$n+$self->{layers}->[$layer]+$self->{layers}->[$layer+1]);
    }
        
        
    # See POD for usage
    sub run {
    	my $self	=	shift;
    	my $inputs	=	shift;
    	my $const	=	$self->{const};
    	#my $start	=	new Benchmark;
    	$inputs		=	$self->crunch($inputs) if($inputs == 0);
    	no strict 'refs';
    	for my $x (0..$#{$inputs}) {
    		d("inputing $inputs->[$x] at index $x with ID $self->{input}->{IDs}->[$x].\n",1);
    		$self->{mesh}->[$x]->input($inputs->[$x]+$const,$self->{input}->{IDs}->[$x]);
    	}
    	if($#{$inputs}<$self->{inputs}-1) {
	    	for my $x ($#{$inputs}+1..$self->{inputs}-1) {
	 	    	d("inputing 1 at index $x with ID $self->{input}->{IDs}->[$x].\n",1);
	    		$self->{mesh}->[$x]->input(1,$self->{input}->{IDs}->[$x]);
	    	}
	    }
    	#$self->{benchmark} = timestr(timediff(new Benchmark, $start));
    	return $self->{output}->get_outputs();
    }    
    
    # See POD for usage
    sub run_uc {
    	$_[0]->uncrunch(run(@_));
    }

	# See POD for usage
	sub learn {
    	my $self	=	shift;
    	my $inputs	=	shift;
    	my $outputs	=	shift;
    	my %args	=	@_;
    	my $inc		=	$args{inc} || 0.1;
    	my $max     =   $args{max} || 1024;               
    	
		my $error   = 	($args{error}>-1 && defined $args{error}) ? $args{error} : -1;
  		my $dinc	=	0.0001;
		my $diff	=	100;
		my $start	=	new Benchmark;
		$inputs		=	$self->crunch($inputs)  if($inputs == 0);
		$outputs	=	$self->crunch($outputs) if($outputs == 0);
		my ($flag,$ldiff,$cdiff,$_mi,$loop,$y);
		while(!$flag && ($max ? $loop<$max : 1)) {
    		my $b	=	new Benchmark;
    		my $got	=	$self->run($inputs);
    		$diff 	=	pdiff($got,$outputs);
		    $flag	=	1;
    		
		    if(!($error>-1 ? $diff>$error : 1)) {
				$flag=1;
				last;
			}
			
			$inc   -= ($dinc*$diff);
			
			if($diff eq $ldiff) {
				$cdiff++;
				$inc += ($dinc*$diff)+($dinc*$cdiff*10);
			} else {
				$cdiff=0;
			}
			$ldiff = $diff;
			
    		for my $x (0..$self->{outputs}-1) {
    			my $a	=	$got->[$x];
    			my $b	=	$outputs->[$x];
    			d("got: $a, wanted: $b\n",2);
    			if ($a != 	$b) {
    				$flag	=	0;
    				$y 		=	$self->{total_nodes}-$self->{outputs}+$x;
    				$self->{mesh}->[$y]->adjust_weight((($a<$b)?1:-1)*$inc);
   				}
   			}
   			
   			$loop++;
   			d("Current Error: $diff, Loop: $loop, Benchmark: ".timestr(timediff(new Benchmark,$b))."\n",4);
   			d("Actual:\n",4);	
   			join_cols($got,($self->{col_width})?$self->{col_width}:5) if(d()==4);
   			d("Target:\n",4);	
   			join_cols($outputs,($self->{col_width})?$self->{col_width}:5) if(d()==4);
   			
   		}  
   		my $str = "Learning took $loop loops and ".timestr(timediff(new Benchmark,$start))."\n";
   		d($str,3); $self->{benchmark} = "$loop loops and ".timestr(timediff(new Benchmark,$start))."\n";
   		return $str;
   	}


	# See POD for usage
	sub learn_set {
		my $self	=	shift;
		my $data	=	shift;
		my %args	=	@_;
		my $len		=	$#{$data}/2-1;
		my $inc		=	$args{inc};
		my $max		=	$args{max};
	    my $error	=	$args{error};
	    my $p		=	(defined $args{p})	?$args{p}	 :1;
	    my $row		=	(defined $args{row})?$args{row}+1:1;
		for my $x (0..$len) {
			my $str = $self->learn( $data->[$x*2],
					  		  		$data->[$x*2+1],
					    			inc=>$inc,
					    			max=>$max,
					    			error=>$error);
		}
			
		if ($p) {
			return pdiff($data->[$row],$self->run($data->[$row-1]));
		} else {
			return $data->[$row]->[0]-$self->run($data->[$row-1])->[0];
		}
	}
	
	# Save entire network state to disk.
	sub save {
		my $self	=	shift;
		my $file	=	shift;
		no strict 'refs';
		
		open(FILE,">$file");
	    
	    print FILE "header=$AI::NeuralNet::Mesh::ID\n";
	   	
		print FILE "total_layers=$self->{total_layers}\n";
		print FILE "total_nodes=$self->{total_nodes}\n";
	    print FILE "nodes=$self->{nodes}\n";
	    print FILE "inputs=$self->{inputs}\n";
	    print FILE "outputs=$self->{outputs}\n";
	    print FILE "layers=",(($self->{layers})?join(',',@{$self->{layers}}):''),"\n";
	    
	    print FILE "rand=$self->{random}\n";
	    print FILE "const=$self->{const}\n";
	    print FILE "cw=$self->{col_width}\n";
		print FILE "crunch=$self->{_crunched}->{_length}\n";
		print FILE "rA=$self->{rA}\n";
		print FILE "rB=$self->{rB}\n";
		print FILE "rS=$self->{rS}\n";
		print FILE "rRef=",(($self->{rRef})?join(',',@{$self->{rRef}}):''),"\n";
			
		for my $a (0..$self->{_crunched}->{_length}-1) {
			print FILE "c$a=$self->{_crunched}->{list}->[$a]\n";
		}
	
		my $n = 0;
		for my $x (0..$self->{total_layers}) {
			for my $y (0..$self->{layers}->[$x]-1) {
			    my $w='';
				for my $z (0..$self->{layers}->[$x-1]-1) {
					$w.="$self->{mesh}->[$n]->{_inputs}->[$z]->{weight},";
				}
				print FILE "n$n=$w$self->{mesh}->[$n]->{activation},$self->{mesh}->[$n]->{threshold},$self->{mesh}->[$n]->{mean}\n";
				$n++;
			}
		}
		
	    close(FILE);
	    
	    if(!(-f $file)) {
	    	$self->{error} = "Error writing to \"$file\".";
	    	return undef;
	    }
	    
	    return $self;
	}
        
	# Load entire network state from disk.
	sub load {
		my $self		=	shift;
		my $file		=	shift;  
		my $load_flag   =	shift;
		
	    if(!(-f $file)) {
	    	$self->{error} = "File \"$file\" does not exist.";
	    	return undef;
	    }
	    
	    open(FILE,"$file");
	    my @lines=<FILE>;
	    close(FILE);
	    
	    my %db;
	    for my $line (@lines) {
	    	chomp($line);
	    	my ($a,$b) = split /=/, $line;
	    	$db{$a}=$b;
	    }
	    
	    if(!$db{"header"}) {
	    	$self->{error} = "Invalid format.";
	    	return undef;
	    }
	    
	    return $self->load_old($file) if($self->version($db{"header"})<0.21);
	    
	    if($load_flag) {
		    undef $self;
	        $self = AI::NeuralNet::Mesh->new([split(',',$db{layers})]);
		} else {
			$self->{inputs}			= $db{inputs};
		    $self->{nodes}			= $db{nodes};
			$self->{outputs}		= $db{outputs};
			$self->{layers} 		= [split(',',$db{layers})];
			$self->{total_layers}	= $db{total_layers};
			$self->{total_nodes}	= $db{total_nodes};
		}
		
	    # Load variables
	    $self->{random}		= $db{"rand"};
	    $self->{const}		= $db{"const"};
        $self->{col_width}	= $db{"cw"};
	    $self->{rA}			= $db{"rA"};
		$self->{rB}			= $db{"rB"};
		$self->{rS}			= $db{"rS"};
		$self->{rRef}		= [split /\,/, $db{"rRef"}];
		
	   	$self->{_crunched}->{_length}	=	$db{"crunch"};
		
		for my $a (0..$self->{_crunched}->{_length}-1) {
			$self->{_crunched}->{list}->[$a] = $db{"c$a"}; 
		}
		
		$self->_init();
	    
		my $n = 0;
		for my $x (0..$self->{total_layers}) {
			for my $y (0..$self->{layers}->[$x]-1) {
			    my @l = split /\,/, $db{"n$n"};
				for my $z (0..$self->{layers}->[$x-1]-1) {
					$self->{mesh}->[$n]->{_inputs}->[$z]->{weight} = $l[$z];
				}
				my $z = $self->{layers}->[$x-1];
				$self->{mesh}->[$n]->{activation} = $l[$z];
				$self->{mesh}->[$n]->{threshold}  = $l[$z+1];
				$self->{mesh}->[$n]->{mean}       = $l[$z+2];
				$n++;
			}
		}
		
		return $self;
	}
	
	# Load entire network state from disk.
	sub load_old {
		my $self		=	shift;
		my $file		=	shift;  
		my $load_flag   =	shift;
		
	    if(!(-f $file)) {
	    	$self->{error} = "File \"$file\" does not exist.";
	    	return undef;
	    }
	    
	    open(FILE,"$file");
	    my @lines=<FILE>;
	    close(FILE);
	    
	    my %db;
	    for my $line (@lines) {
	    	chomp($line);
	    	my ($a,$b) = split /=/, $line;
	    	$db{$a}=$b;
	    }
	    
	    if(!$db{"header"}) {
	    	$self->{error} = "Invalid format.";
	    	return undef;
	    }
	    
	    if($load_flag) {
		    undef $self;
	
			# Create new network
			$self = AI::NeuralNet::Mesh->new($db{"layers"},
		    			 				 	 $db{"nodes"},
		    						      	 $db{"outputs"});
		} else {
			$self->{total_layers}	=	$db{"layers"};
			$self->{nodes}			=	$db{"nodes"};
			$self->{outputs}		=	$db{"outputs"};
			$self->{inputs}			=	$db{"nodes"};
			#$self->{total_nodes}	=	$db{"total"};
		}
		
	    # Load variables
	    $self->{random}		= $db{"rand"};
	    $self->{const}		= $db{"const"};
        $self->{col_width}	= $db{"cw"};
	    $self->{rA}			= $db{"rA"};
		$self->{rB}			= $db{"rB"};
		$self->{rS}			= $db{"rS"};
		$self->{rRef}		= [split /\,/, $db{"rRef"}];
		
	   	$self->{_crunched}->{_length}	=	$db{"crunch"};
		
		for my $a (0..$self->{_crunched}->{_length}-1) {
			$self->{_crunched}->{list}->[$a] = $db{"c$a"}; 
		}
		
	
		$self->_init();
	    
	    my $nodes	=	$self->{nodes};
	   	my $outputs	=	$self->{outputs};
	   	my $tmp		=	$self->{total_nodes};
	   	my $div 	=	intr($nodes/$outputs);

		# Load input and hidden
		for my $a (0..$tmp-1) {
			my @l = split /\,/, $db{"n$a"};
			for my $b (0..$nodes-1) {
				$self->{mesh}->[$a]->{_inputs}->[$b]->{weight} = $l[$b];
			}                  
		}
	     
		# Load output layer
		for my $x (0..$outputs-1) {
			my @l = split /\,/, $db{"n".($tmp+$x)};
			for my $y (0..$div-1) {
				$self->{mesh}->[$tmp+$x]->{_inputs}->[$y]->{weight} = $l[$y];
		 	}
		} 
		
		return $self;
	}

	# Dumps the complete weight matrix of the network to STDIO
	sub show {
		my $self	=	shift;
		my $n 		=	0;    
		no strict 'refs';
		for my $x (0..$self->{total_layers}) {
			for my $y (0..$self->{layers}->[$x]-1) {
				for my $z (0..$self->{layers}->[$x-1]-1) {
					print "$self->{mesh}->[$n]->{_inputs}->[$z]->{weight},";
				}
				$n++;
			}
			print "\n";
		}
	}
	  
	# Set the activation type of a specific layer.
	# usage: $net->activation($layer,$type);
	# $type can be: "linear", "sigmoid", "sigmoid_2".
	# You can use "sigmoid_1" as a synonym to "sigmoid". 
	# Type can also be a CODE ref, ( ref($type) eq "CODE" ).
	# If $type is a CODE ref, then the function is called in this form:
	# 	$output	= &$type($sum_of_inputs,$self);
	# The code ref then has access to all the data in that node (thru the
	# blessed refrence $self) and is expected to return the value to be used
	# as the output for that node. The sum of all the inputs to that node
	# is already summed and passed as the first argument.
	sub activation {
		my $self	=	shift;
		my $layer	=	shift || 0;
		my $value	=	shift || 'linear';
		my $n 		=	0;    
		no strict 'refs';
		for(0..$layer-1){$n+=$self->{layers}->[$_]}
		for($n..$n+$self->{layers}->[$layer]-1) {
			$self->{mesh}->[$_]->{activation} = $value; 
		}
	}
	
	# Applies an activation type to a specific node
	sub node_activation {
		my $self	=	shift;
		my $layer	=	shift || 0;
		my $node	=	shift || 0;
		my $value	=	shift || 'linear';
		my $n 		=	0;    
		no strict 'refs';
		for(0..$layer-1){$n+=$self->{layers}->[$_]}
		$self->{mesh}->[$n+$node]->{activation} = $value; 
	}
	
	# Set the activation threshold for a specific layer.
	# Only applicable if that layer uses "sigmoid" or "sigmoid_2"
	# usage: $net->threshold($layer,$threshold);
	sub threshold {
		my $self	=	shift;
		my $layer	=	shift || 0;
		my $value	=	shift || 0.5; 
		my $n		=	0;
		no strict 'refs';
		for(0..$layer-1){$n+=$self->{layers}->[$_]}
		for($n..$n+$self->{layers}->[$layer]-1) {
			$self->{mesh}->[$_]->{threshold} = $value;
		}
	}
	
	# Applies a threshold to a specific node
	sub node_threshold {
		my $self	=	shift;
		my $layer	=	shift || 0;
		my $node	=	shift || 0;
		my $value	=	shift || 0.5; 
		my $n		=	0;
		no strict 'refs';
		for(0..$layer-1){$n+=$self->{layers}->[$_]}
		$self->{mesh}->[$n+$node]->{threshold} = $value;
	}
	
	# Set mean (avg.) flag for a layer.
	# usage: $net->mean($layer,$flag);
	# If $flag is true, it enables finding the mean for that layer,
	# If $flag is false, disables mean.
	sub mean {
		my $self	=	shift;
		my $layer	=	shift || 0;
		my $value	=	shift || 0;
		my $n		=	0;
		no strict 'refs';
		for(0..$layer-1){$n+=$self->{layers}->[$_]}
		for($n..$n+$self->{layers}->[$layer]-1) {
			$self->{mesh}->[$_]->{mean} = $value;
		}
	}
	
	  
	# Returns a pcx object
	sub load_pcx {
		my $self	=	shift;
		my $file	=	shift;
		eval('use PCX::Loader');
		if(@_) {
			$self->{error}="Cannot load PCX::Loader module: @_";
			return undef;
		}
		return PCX::Loader->new($self,$file);
	}	
	
	# Crunch a string of words into a map
	sub crunch {
		my $self	=	shift;
		my @ws 		=	split(/[\s\t]/,shift);
		my (@map,$ic);
		for my $a (0..$#ws) {
			$ic=$self->crunched($ws[$a]);
			if(!defined $ic) {
				$self->{_crunched}->{list}->[$self->{_crunched}->{_length}++]=$ws[$a];
				@map[$a]=$self->{_crunched}->{_length};
			} else {
				@map[$a]=$ic;
            }
		}
		return \@map;
	}
	
	# Finds if a word has been crunched.
	# Returns undef on failure, word index for success.
	sub crunched {
		my $self	=	shift;
		for my $a (0..$self->{_crunched}->{_length}-1) {
			return $a+1 if($self->{_crunched}->{list}->[$a] eq $_[0]);
		}
		$self->{error} = "Word \"$_[0]\" not found.";
		return undef;
	}
	
	# Alias for crunched(), above
	sub word { crunched(@_) }
	
	# Uncrunches a map (array ref) into an array of words (not an array ref) 
	# and returns array
	sub uncrunch {
		my $self	=	shift;
		my $map = shift;
		my ($c,$el,$x);
		foreach $el (@{$map}) {
			$c .= $self->{_crunched}->{list}->[$el-1].' ';
		}
		return $c;
	}
	
	# Sets/gets randomness facter in the network. Setting a value of 0 
	# disables random factors.
	sub random {
		my $self	=	shift;
		my $rand	=	shift;
		return $self->{random}	if(!(defined $rand));
		$self->{random}	=	$rand;
	}
	
	# Sets/gets column width for printing lists in debug modes 1,3, and 4.
	sub col_width {
		my $self	=	shift;
		my $width	=	shift;
		return $self->{col_width}	if(!$width);
		$self->{col_width}	=	$width;
	} 

	# Sets/gets run const. facter in the network. Setting a value of 0 
	# disables run const. factor. 
	sub const {
		my $self	=	shift;
		my $const	=	shift;
		return $self->{const}	if(!(defined $const));
		$self->{const}	=	$const;
	}
	
	# Sets/Removes value ranging
	# NOTE: This is disabled in this version,
	# it has no effect.
	sub range {
		my $self	=	shift;
		my $ref		=	shift;
		my $b		=	shift;
		if(substr($ref,0,5) ne "ARRAY") {
			if(($ref == 0) && (!defined $b)) {
				$ref	= $self->crunch($ref);
			} else {
    			my $a	= $ref;
    			$a		= $self->crunch($a)->[0] if($a == 0);
				$b		= $self->crunch($b)->[0] if($b == 0);
				$_[++$#_] = $a;
    			$_[++$#_] = $b;
    			$ref	= \@_;
			}
		}
		my $rA		=	0;
		my $rB		=	$#{$ref};
		my $rS		=	0;
		if(!$rA && !$rB) {
			$self->{rA}=$self->{rB}=-1;
			$self->{error}="Internal range error.";
			return undef;
		}
		if($rB<$rA){my $t=$rA;$rA=$rB;$rB=$t};
		$self->{rA}		=	$rA;
		$self->{rB}		=	$rB;
		$self->{rS}		=	$rS if($rS);
		$self->{rRef}	=	$ref;
		return $ref;
	}                                                                        	
	
	# Used internally to scale outputs to fit range
	sub _scale_outputs {
		my $self	=	shift;  
		my $in		=	shift;
		my $rA		=	$self->{rA};
		my $rB		=	$self->{rB};
		my $rS		=	$self->{rS};
		my $r		=	$rB;#-$rA+1;
		my $l		=	$self->{outputs}-1;
		my $out 	=	[];
		
		# I've disabled scaling the outputs in this Mesh because they 
		# never seem to be able to learn correctly. Maybe later...
		return $in;
		
 		return $in if(!$rA && !$rB);
		# Adjust for a maximum outside what we have seen so far
		for my $i (0..$l) {
			$rS=$in->[$i]+1 if($in->[$i]+1>$rS);
		}
		# Loop through, convert values to percentage of maximum, then multiply
		# percentage by range and add to base of range to get finaly value
		for my $i (0..$l) {
			$rS=1 if(!$rS);
			my $t = intr((($rS-$in->[$i])/$rS)*$rB);
			print "t:$t,in:$in->[$i],rB:$rB,rS:$rS,minus:",($rS-$in->[$i]),"\n";
			$out->[$i] = $self->{rRef}->[$t];
		}
		$self->{rS}=$rS;
		return $out;
	}
	
	# Return benchmark time from last learn() operation.
	sub benchmark {
		shift->{benchmarked};
	}
	
	# Same as benchmark()
	sub benchmarked {
		benchmark(shift);
	}
	
	# Return the last error in the mesh, or undef if no error.
	sub error {
		my $self = shift;
		return undef if !$self->{error};
		chomp($self->{error});
		return $self->{error}."\n";
	}
	
	# Rounds a floating-point to an integer with int() and sprintf()
	sub intr  {
    	shift if(substr($_[0],0,4) eq 'AI::');
      	try   { return int(sprintf("%.0f",shift)) }
      	catch { return 0 }
	}
    
	# Used to format array ref into columns
	# Usage: 
	#	join_cols(\@array,$row_length_in_elements,$high_state_character,$low_state_character);
	# Can also be called as method of your neural net.
	# If $high_state_character is null, prints actual numerical values of each element.
	sub join_cols {
		no strict 'refs';
		shift if(substr($_[0],0,4) eq 'AI::'); 
		my $map		=	shift;
		my $break   =	shift;
		my $a		=	shift;
		my $b		=	shift;
		my $x;
		foreach my $el (@{$map}) { 
			my $str = ((int($el))?$a:$b);
			$str=$el."\0" if(!$a);
			print $str;	$x++;
			if($x>$break-1) { print "\n"; $x=0;	}
		}
		print "\n";
	}
	
	# Returns percentage difference between all elements of two
	# array refs of exact same length (in elements).
	# Now calculates actual difference in numerical value.
	sub pdiff {
		no strict 'refs';
		shift if(substr($_[0],0,4) eq 'AI::'); 
		my $a1	=	shift;
		my $a2	=	shift;
		my $a1s	=	$#{$a1};
		my $a2s	=	$#{$a2};
		my ($a,$b,$diff,$t);
		$diff=0;
		for my $x (0..$a1s) {
			$a = $a1->[$x]; $b = $a2->[$x];
			if($a!=$b) {
				if($a<$b){$t=$a;$a=$b;$b=$t;}
				$a=1 if(!$a); $diff+=(($a-$b)/$a)*100;
			}
		}
		$a1s = 1 if(!$a1s);
		return sprintf("%.20f",($diff/$a1s));
	}
	
	# Returns $fa as a percentage of $fb
	sub p {
		shift if(substr($_[0],0,4) eq 'AI::'); 
		my ($fa,$fb)=(shift,shift);
		sprintf("%.3f",((($fb-$fa)*((($fb-$fa)<0)?-1:1))/$fa)*100);
	}
	
	# Returns the index of the element in array REF passed with the highest comparative value
	sub high {
		shift if(substr($_[0],0,4) eq 'AI::'); 
		my $ref1 = shift; my ($el,$len,$tmp); $tmp=0;
		foreach $el (@{$ref1}) { $len++ }
		for my $x (0..$len-1) { $tmp = $x if($ref1->[$x] > $ref1->[$tmp]) }
		return $tmp;
	}
	
	# Returns the index of the element in array REF passed with the lowest comparative value
	sub low {
		shift if(substr($_[0],0,4) eq 'AI::'); 
		my $ref1 = shift; my ($el,$len,$tmp); $tmp=0;
		foreach $el (@{$ref1}) { $len++ }
		for my $x (0..$len-1) { $tmp = $x if($ref1->[$x] < $ref1->[$tmp]) }
		return $tmp;
	}  
	
		
1;

package AI::NeuralNet::Mesh::node;
	
	use strict;

	# Node constructor
	sub new {
		my $type		=	shift;
		my $self		={ 
			_parent		=>	shift,
			_inputs		=>	[],
			_outputs	=>	[]
		};
		bless $self, $type;
	}

	# Receive inputs from other nodes, and also send
	# outputs on.	
	sub input {
		my $self	=	shift;
		my $input	=	shift;
		my $from_id	=	shift;
		
		$self->{_inputs}->[$from_id]->{value} = $input * $self->{_inputs}->[$from_id]->{weight};
		$self->{_inputs}->[$from_id]->{fired} = 1;
		
		$self->{_parent}->d("got input $input from id $from_id, weighted to $self->{_inputs}->[$from_id]->{value}.\n",1);
		
		my $flag	=	1;
		for my $x (0..$self->{_inputs_size}-1) { $flag = 0 if(!$self->{_inputs}->[$x]->{fired}) }
		if ($flag) {
			$self->{_parent}->d("all inputs fired for $self.\n",1);
			my $output	=	0;   
			
			# Sum
			for my $i (@{$self->{_inputs}}) {                        
				$output += $i->{value};
			}
		
			# Handle activations, thresholds, and means
			$output	   /=  $self->{_inputs_size} if($self->{flag_mean});
			$output    += (rand()*$self->{_parent}->{random});
			$output		= ($output>=$self->{threshold})?1:0 if(($self->{activation} eq "sigmoid") || ($self->{activation} eq "sigmoid_1"));
			if($self->{activation} eq "sigmoid_2") {
				$output =  1 if($output >$self->{threshold});
				$output = -1 if($output <$self->{threshold});
				$output =  0 if($output==$self->{threshold});
			}
			
			# Handle CODE refs
			$output = &{$self->{activation}}($output,$self) if(ref($self->{activation}) eq "CODE");
			
			# Send output
			for my $o (@{$self->{_outputs}}) { $o->{node}->input($output,$o->{from_id}) }
		} else {
			$self->{_parent}->d("all inputs have NOT fired for $self.\n",1);
		}
	}

	sub add_input_node {
		my $self	=	shift;
		my $node	=	shift;
		my $i		=	$self->{_inputs_size} || 0;
		$self->{_inputs}->[$i]->{node}	 = $node;
		$self->{_inputs}->[$i]->{value}	 = 0;
		$self->{_inputs}->[$i]->{weight} = 1;
		$self->{_inputs}->[$i]->{fired}	 = 0;
		$self->{_inputs_size} = ++$i;
		return $i-1;
	}
	
	sub add_output_node {
		my $self	=	shift;
		my $node	=	shift;
		my $i		=	$self->{_outputs_size} || 0;
		$self->{_outputs}->[$i]->{node}		= $node;
		$self->{_outputs}->[$i]->{from_id}	= $node->add_input_node($self);
		$self->{_outputs_size} = ++$i;
		return $i-1;
	}     
	
	sub adjust_weight {
		my $self	=	shift;
		my $inc		=	shift;
		for my $i (@{$self->{_inputs}}) {
			$i->{weight} += $inc * $i->{weight};
			$i->{node}->adjust_weight($inc) if($i->{node});
		}
	}

1;	
	
# Internal usage, prevents recursion on empty nodes.
package AI::NeuralNet::Mesh::cap;
	sub new     { bless {}, shift }
	sub input           {}
	sub adjust_weight   {}
	sub add_output_node {}
	sub add_input_node  {}
1;

# Internal usage, collects data from output layer.
package AI::NeuralNet::Mesh::output;
	
	use strict;
	
	sub new {
		my $type		=	shift;
		my $self		={ 
			_parent		=>	shift,
			_inputs		=>	[],
		};
		bless $self, $type;
	}
	
	sub add_input_node {
		my $self	=	shift;
		return (++$self->{_inputs_size})-1;
	}
	
	sub input {
		my $self	=	shift;
		my $input	=	shift;
		my $from_id	=	shift;
		$self->{_parent}->d("GOT INPUT [$input] FROM [$from_id]\n",1);
		$self->{_inputs}->[$from_id] = $self->{_parent}->intr($input);
	}
	
	sub get_outputs {
		my $self	=	shift;
		return $self->{_parent}->_scale_outputs($self->{_inputs});
	}

1;
                                       
__END__

=head1 NAME

AI::NeuralNet::Mesh - An optimized, accurate neural network Mesh.

=head1 SYNOPSIS
    
	use AI::NeuralNet::Mesh;

    # Create a mesh with 2 layers, 2 nodes/layer, and one output node.
	my $net = new AI::NeuralNet::Mesh(2,2,1);
	
	# Teach the network the AND function
	$net->learn([0,0],[0]);
	$net->learn([0,1],[0]);
	$net->learn([1,0],[0]);
	$net->learn([1,1],[1]);
	
	# Present it with two test cases
	my $result_bit_1 = $net->run([0,1])->[0];
	my $result_bit_2 = $net->run([1,1])->[0];
	
	# Display the results
	print "AND test with inputs (0,1): $result_bit_1\n";
	print "AND test with inputs (1,1): $result_bit_2\n";
	

=head1 VERSION & UPDATES

This is version B<0.31>, the second release of this module. 

In this version, I have included three major features. Also in this
release I have included two minor fixes which increase the learning speed
of networks. I also fixed a bug in the load_pcx() method which prevented it
from loading the PCX::Loader module correctly. This version also has the ability
to have negative weights in the network.

The major features added are:

=item LAYER SIZES

Rodin Porrata once suggested it would be good to have control over
each layer's node size. Well, Rodin, here you go. Each layer can 
have a custom number of nodes, which you can set in two ways, detailed
in the new() constructor, below. Layer sizes are preserved across load()
and save() calls.

=item LAYER EXTENSION

With the ability to have custom layer sizes, I have also included the ability to 
extend layer sizes after network construction. You can add nodes with extend() or
extend_layer() after the network is constructed or loaded.

=item CUSTOM NODE ACTIVATION

Ahh, and another treat. You can choose from one of four activation functions
and set the activation function by layer, or you can even set each individual
node to a seperate activation function. Possible activation types are: C<linear>
(simply transfer sum of inputs as output), C<sigmoid> (also called C<sigmoid_1>) (0 or 1, threshold based),
 C<sigmoid_2> (-1,0,1, threshold based), or user specified (passed as a CODE ref.)
 You can also customize threshold levels on a per-layer, or per-node basis.
 
For more information on setting activation and threshold levels, see the new() constructor,
or any of the activation() or threshold() methods.

=head1 DESCRIPTION

AI::NeuralNet::Mesh is an optimized, accurate neural network Mesh.
It was designed with accruacy and speed in mind. This is a neural
net simulator similar to AI::NeuralNet::BackProp, but with several
important differences. The two APIs are the same, that of this module
and ::BackProp, so any scripts that use ::BackProp, should be able
to use this module without (almost) any changes in your code. (The 
only changes needed will be to change the "use" line and the "new" 
constructor line to use ::Mesh instead of ::BackProp.)

This is a module complete, from-scratch re-write of the Perl module 
AI::NeuralNet::BackProp. It a method of learning similar to 
back propogation, yet with a few custom modifications, includeding
a specialized output layer, as well as a better descent model for 
learning. 

Almost all of the notes and description in AI::NeuralNet::BackProp
apply to this module, yet the differences I will detail below. I
also have included a complete working function refrence here, with
the updates added.


=head1 WHATS DIFFERENT FROM C<::BackProp>?

=head2 MESH CONNECTIONS

In AI::NeuralNet::BackProp, the neurons would be connected like this:


     output
     /  \
    O    O
    |\  /|
    | \/ |
    | /\ |
    |/  \|
    O    O
     \  /
    input
     
     
In this module, I have made a couple of important changes to the connection
map. Consider this digram (This has 2 layers, 2 nodes/layer, 1 output node):
       
 
 data collector 
       ^
       |
       O       <-- OUTPUT LAYER
      / \  
     /   \
    O     O    <-- LAYER 2
    |\   /|
    | \ / |
    | / \ |
    |/   \|
    O     O    <-- LAYER 1
    |     | 
    ^     ^
  input array
    
    
The mesh model includes an extra output "layer" above the final layer 
specified in the constructor. If the constructor had specified 2 layers,
2 nodes/layer, and B<2> output nodes, then the mesh would look like this:


 data collector 
    ^     ^
    |     |
    O     O    <-- OUTPUT LAYER
    |     |
    |     |
    O     O    <-- LAYER 2
    |\   /|
    | \ / |
    | / \ |
    |/   \|
    O     O    <-- LAYER 1
    |     | 
    ^     ^
  input array
    

As you can see, the mesh creator adds one node in the output layer for
every node called for in the constructor. This adds an node
for that output, allowing better accuracy in the network, 
whereas in AI::NeuralNet::BackProp the output nodes were not allowed to be 
weighted.

=head2 LEARNING STYLE

In this module I have included a more accurate form of "learning" for the
mesh. This form preforms descent toward a local error minimum (0) on a 
directional delta, rather than the desired value for that node. This allows
for better, and more accurate results with larger datasets. This module also
uses a simpler recursion technique which, suprisingly, is more accurate than
the original technique that I used in AI::NeuralNet::BackProp.

By way of accuracy example, the included example script "examples/ex_dow.pl",
upon the third learning loop (using AI::NeuralNet::BackProp), would almost
always report forgetfulness around 25.00000% (rounded to five decimals), 
whereas when running the same example and the same example code with 
AI::NeuralNet::Mesh and only B<one> learning loop, it reports forgetfulness 
of less than I<2.00227>! Over twenty-two percent increase in accuracy on one 
script alone.

The learning is also speed up immensly. Whereas the above mentioned script 
often take up to a half hour or more on my systems to learn the example data 
with the old AI::NeuralNet::BackProp module, it now (with this module) takes 
less than I<forty seconds> to learn the data set (one loop).

=head1 METHODS

=item AI::NeuralNet::Mesh->new();

There are four ways to construct a new network with new(). Each is detailed below.

P.S. Don't worry, the old C<new($layers, $nodes [, $outputs])> still works like always!

=item AI::NeuralNet::Mesh->new($layers, $nodes [, $outputs]);

Returns a newly created neural network from an C<AI::NeuralNet::Mesh>
object. The network will have C<$layers> number of layers in it
and it will have C<$nodes> number of nodes per layer.

There is an optional parameter of $outputs, which specifies the number
of output neurons to provide. If $outputs is not specified, $outputs
defaults to equal $size. 


=item AI::NeuralNet::Mesh->new($file);

This will automatically create a new network from the file C<$file>. It will
return undef if the file was of an incorrect format or non-existant. Otherwise,
it will return a blessed refrence to a network completly restored from C<$file>.

=item AI::NeuralNet::Mesh->new(\@layer_sizes);

This constructor will make a network with the number of layers corresponding to the length
in elements of the array ref passed. Each element in the array ref passed is expected
to contain an integer specifying the number of nodes (neurons) in that layer. The first
layer ($layer_sizes[0]) is to be the input layer, and the last layer in @layer_sizes is to be
the output layer.

Example:

	my $net = AI::NeuralNet::Mesh->new([2,3,1]);
	

Creates a network with 2 input nodes, 3 hidden nodes, and 1 output node.


=item AI::NeuralNet::Mesh->new(\@array_of_hashes);

Another dandy constructor...this is my favorite. It allows you to tailor the number of layers,
the size of the layers, the activation type (you can even add anonymous inline subs with this one),
and even the threshold, all with one array ref-ed constructor.

Example:

	my $net = AI::NeuralNet::Mesh->new([
	    {
		    nodes        => 2,
		    activation   => linear
		},
		{
		    nodes        => 3,
		    activation   => sub {
		        my $sum  =  shift;
		        return $sum + rand()*1;
		    }
		},
		{
		    nodes        => 1,
		    activation   => sigmoid,
		    threshold    => 0.75
		}
	]);
	
	
Interesting, eh? What you are basically passing is this:

	my @info = ( 
		{ },
		{ },
		{ },
		...
	);

You are passing an array ref who's each element is a hash refrence. Each
hash refrence, or more precisely, each element in the array refrence you are passing
to the constructor, represents a layer in the network. Like the constructor above,
the first element is the input layer, and the last is the output layer. The rest are
hidden layers.

Each hash refrence is expected to have AT LEAST the "nodes" key set to the number
of nodes (neurons) in that layer. The other two keys are optional. If "activation" is left
out, it defaults to "linear". If "threshold" is left out, it defaults to 0.50.

The "activation" key can be one of four values:

	linear                    ( simply use sum of inputs as output )
	sigmoid    [ sigmoid_1 ]  ( only positive sigmoid )
	sigmoid_2                 ( positive / 0 /negative sigmoid )
	\&code_ref;

"sigmoid_1" is an alias for "sigmoid". 

The code ref option allows you to have a custom activation function for that layer.
The code ref is called with this syntax:

	$output = &$code_ref($sum_of_inputs, $self);
	
The code ref is expected to return a value to be used as the output of the node.
The code ref also has access to all the data of that node through the second argument,
a blessed hash refrence to that node.

Three of the activation syntaxes are shown in the first constructor above, the "linear",
"sigmoid" and code ref types.

You can also set the activation and threshold values after network creation with the
activation() and threshold() methods. 

	



=item $net->learn($input_map_ref, $desired_result_ref [, options ]);

This will 'teach' a network to associate an new input map with a desired 
result. It will return a string containg benchmarking information. 

You can also specify strings as inputs and ouputs to learn, and they will be 
crunched automatically. Example:

	$net->learn('corn', 'cob');
	
	
Note, the old method of calling crunch on the values still works just as well.	

The first two arguments may be array refs (or now, strings), and they may be 
of different lengths.

Options should be written on hash form. There are three options:
	 
	 inc	=>	$learning_gradient
	 max	=>	$maximum_iterations
	 error	=>	$maximum_allowable_percentage_of_error
	 

C<$learning_gradient> is an optional value used to adjust the weights of the internal
connections. If $learning_gradient is ommitted, it defaults to 0.10.
 
C<$maximum_iterations> is the maximum numbers of iteration the loop should do.
It defaults to 1024.  Set it to 0 if you never want the loop to quit before
the pattern is perfectly learned.

C<$maximum_allowable_percentage_of_error> is the maximum allowable error to have. If 
this is set, then learn() will return when the perecentage difference between the
actual results and desired results falls below $maximum_allowable_percentage_of_error.
If you do not include 'error', or $maximum_allowable_percentage_of_error is set to -1,
then learn() will not return until it gets an exact match for the desired result OR it
reaches $maximum_iterations.


=item $net->learn_set(\@set, [ options ]);

This takes the same options as learn() (learn_set() uses learn() internally) 
and allows you to specify a set to learn, rather than individual patterns. 
A dataset is an array refrence with at least two elements in the array, 
each element being another array refrence (or now, a scalar string). For 
each pattern to learn, you must specify an input array ref, and an ouput 
array ref as the next element. Example:
	
	my @set = (
		# inputs        outputs
		[ 1,2,3,4 ],  [ 1,3,5,6 ],
		[ 0,2,5,6 ],  [ 0,2,1,2 ]
	);


Inputs and outputs in the dataset can also be strings.

See the paragraph on measuring forgetfulness, below. There are 
two learn_set()-specific option tags available:

	flag     =>  $flag
	pattern  =>  $row

If "flag" is set to some TRUE value, as in "flag => 1" in the hash of options, or if the option "flag"
is not set, then it will return a percentage represting the amount of forgetfullness. Otherwise,
learn_set() will return an integer specifying the amount of forgetfulness when all the patterns 
are learned. 

If "pattern" is set, then learn_set() will use that pattern in the data set to measure forgetfulness by.
If "pattern" is omitted, it defaults to the first pattern in the set. Example:

	my @set = (
		[ 0,1,0,1 ],  [ 0 ],
		[ 0,0,1,0 ],  [ 1 ],
		[ 1,1,0,1 ],  [ 2 ],  #  <---
		[ 0,1,1,0 ],  [ 3 ]
	);
	
If you wish to measure forgetfulness as indicated by the line with the arrow, then you would
pass 2 as the "pattern" option, as in "pattern => 2".

Now why the heck would anyone want to measure forgetfulness, you ask? Maybe you wonder how I 
even measure that. Well, it is not a vital value that you have to know. I just put in a 
"forgetfulness measure" one day because I thought it would be neat to know. 

How the module measures forgetfulness is this: First, it learns all the patterns 
in the set provided, then it will run the very first pattern (or whatever pattern
is specified by the "row" option) in the set after it has finished learning. It 
will compare the run() output with the desired output as specified in the dataset. 
In a perfect world, the two should match exactly. What we measure is how much that 
they don't match, thus the amount of forgetfulness the network has.

Example (from examples/ex_dow.pl):

	# Data from 1989 (as far as I know..this is taken from example data on BrainMaker)
	my @data = ( 
		#	Mo  CPI  CPI-1 CPI-3 	Oil  Oil-1 Oil-3    Dow   Dow-1 Dow-3   Dow Ave (output)
		[	1, 	229, 220,  146, 	20.0, 21.9, 19.5, 	2645, 2652, 2597], 	[	2647  ],
		[	2, 	235, 226,  155, 	19.8, 20.0, 18.3, 	2633, 2645, 2585], 	[	2637  ],
		[	3, 	244, 235,  164, 	19.6, 19.8, 18.1, 	2627, 2633, 2579], 	[	2630  ],
		[	4, 	261, 244,  181, 	19.6, 19.6, 18.1, 	2611, 2627, 2563], 	[	2620  ],
		[	5, 	276, 261,  196, 	19.5, 19.6, 18.0, 	2630, 2611, 2582], 	[	2638  ],
		[	6, 	287, 276,  207, 	19.5, 19.5, 18.0, 	2637, 2630, 2589], 	[	2635  ],
		[	7, 	296, 287,  212, 	19.3, 19.5, 17.8, 	2640, 2637, 2592], 	[	2641  ] 		
	);
	
	# Learn the set
	my $f = $net->learn_set(\@data, 
					  inc	=>	0.1,	
					  max	=>	500,
					 );
			
	# Print it 
	print "Forgetfullness: $f%";

    
This is a snippet from the example script examples/finance.pl, which demonstrates DOW average
prediction for the next month. A more simple set defenition would be as such:

	my @data = (
		[ 0,1 ], [ 1 ],
		[ 1,0 ], [ 0 ]
	);
	
	$net->learn_set(\@data);
	
Same effect as above, but not the same data (obviously).


=item $net->run($input_map_ref);

NOTE: This is a deviation from the AI::NeuralNet::BackProp API.
In ::BackProp, run() automatically benchmarked itself. In ::Mesh run() does
not do this inorder to speed up learn() by as much as 10-20 seconds on small
data sets. Even larger speed increases are realized on larger data sets.

This method will apply the given array ref at the input layer of the neural network, and
it will return an array ref to the output of the network. run() will now automatically crunch() 
a string given as an input (See the crunch() method for info on crunching).

Example Usage:
	
	my $inputs  = [ 1,1,0,1 ];
	my $outputs = $net->run($inputs);

You can also do this with a string:

	my $outputs = $net->run('cloudy - wind is 5 MPH NW');
	

See also run_uc() below.


=item $net->run_uc($input_map_ref);

This method does the same thing as this code:
	
	$net->uncrunch($net->run($input_map_ref));

All that run_uc() does is that it automatically calls uncrunch() on the output, regardless
of whether the input was crunch() -ed or not.
	


=item $net->range();

NOTE: This is a deviation from the AI::NeuralNet::BackProp API.
::BackProp has range() enabled, ::Mesh does not.

In this module, range() is disabled. It is included as a function stub
to comply with the API established by AI::NeuralNet::BackProp. I have 
included the full code to the two essential parts of range() in the module
file, though. If anyone feels up to it, they can attempt to get range() 
working on their own. If you do get range working, please send me a copy! :-)


=item $net->benchmark();

=item $net->benchmarked();

NOTE: This is a deviation from the AI::NeuralNet::BackProp API.
In ::BackProp, benchmarked() returns benchmark info for last run() call.
In ::Mesh it only will return info for the last learn() call. benchmarked()
is an alias for benchmark() so we don't break any scripts.. 

This returns a benchmark info string for the last learn() call.
It is easily printed as a string, as following:

	print "Last learn() took ",$net->benchmark(),"\n";




=item $net->debug($level)

Toggles debugging off if called with $level = 0 or no arguments. There are four levels
of debugging. 

NOTE: Debugging verbosity has been toned down somewhat from AI::NeuralNet::BackProp,
but level 4 still prints the same amount of information as you were used to. The other
levels, however, are mostly for really advanced use. Not much explanation in the other
levels, but they are included for those of you that feel daring (or just plain bored.)

Level 0 ($level = 0) : Default, no debugging information printed. All printing is 
left to calling script.

Level 1 ($level = 1) : Displays the activity between nodes, prints what values were
received and what they were weighted to.

Level 2 ($level = 2) : Just prints info from the learn() loop, in the form of "got: X, wanted Y"
type of information.   

Level 3 ($level = 3) : I don't think I included any level 3 debugs in this version.

Level 4 ($level = 4) : This level is the one I use most. It is only used during learning. It
displays the current error (difference between actual outputs and the target outputs you
asked for), as well as the current loop number and the benchmark time for the last learn cycle.
Also printed are the actual outputs and the target outputs below the benchmark times.

Toggles debuging off when called with no arguments. 



=item $net->save($filename);

This will save the complete state of the network to disk, including all weights and any
words crunched with crunch() . Also saves the layer size and activations of the network.

NOTE: The only activation type NOT saved is the CODE ref type, which must be set again
after loading.

This uses a simple flat-file text storage format, and therefore the network files should
be fairly portable.

This method will return undef if there was a problem with writing the file. If there is an
error, it will set the internal error message, which you can retrive with the error() method,
below.

If there were no errors, it will return a refrence to $net.


=item $net->load($filename);

This will load from disk any network saved by save() and completly restore the internal
state at the point it was save() was called at.

If the file doesn't exist, or if the file is of an invalid file type, then load() will
return undef. To determine what caused the error, use the error() method, below.

If there were no errors, it will return a refrence to $net.



=item $net->activation($layer,$type);

This sets the activation type for layer C<$layer>.

C<$type> can be one of four values:

	linear                    ( simply use sum of inputs as output )
	sigmoid    [ sigmoid_1 ]  ( only positive sigmoid )
	sigmoid_2                 ( positive / 0 /negative sigmoid )
	\&code_ref;

"sigmoid_1" is an alias for "sigmoid". 

The code ref option allows you to have a custom activation function for that layer.
The code ref is called with this syntax:

	$output = &$code_ref($sum_of_inputs, $self);
	
The code ref is expected to return a value to be used as the output of the node.
The code ref also has access to all the data of that node through the second argument,
a blessed hash refrence to that node.

The activation type for each layer is preserved across load/save calls. 

EXCEPTION: Due to the constraints of Perl, I cannot load/save the actual subs that the code
ref option points to. Therefore, you must re-apply any code ref activation types after a 
load() call.

=item $net->node_activation($layer,$node,$type);

This sets the activation function for a specific node in a layer. The same notes apply
here as to the activation() method above.


=item $net->threshold($layer,$value);

This sets the activation threshold for a specific layer. The threshold only is used
when activation is set to "sigmoid", "sigmoid_1", or "sigmoid_2". 


=item $net->node_threshold($layer,$node,$value);

This sets the activation threshold for a specific node in a layer. The threshold only is used
when activation is set to "sigmoid", "sigmoid_1", or "sigmoid_2".  

=item $net->join_cols($array_ref,$row_length_in_elements,$high_state_character,$low_state_character);

This is more of a utility function than any real necessary function of the package.
Instead of joining all the elements of the array together in one long string, like join() ,
it prints the elements of $array_ref to STDIO, adding a newline (\n) after every $row_length_in_elements
number of elements has passed. Additionally, if you include a $high_state_character and a $low_state_character,
it will print the $high_state_character (can be more than one character) for every element that
has a true value, and the $low_state_character for every element that has a false value. 
If you do not supply a $high_state_character, or the $high_state_character is a null or empty or 
undefined string, it join_cols() will just print the numerical value of each element seperated
by a null character (\0). join_cols() defaults to the latter behaviour.



=item $net->extend(\@array_of_hashes);

This allows you to re-apply any activations and thresholds with the same array ref which
you created a network with. This is useful for re-applying code ref activations after a load()
call without having to type the code ref twice.

You can also specify the extension in a simple array ref like this:

	$net->extend([2,3,1]);
	
Which will simply add more nodes if needed to set the number of nodes in each layer to their 
respective elements. This works just like the respective new() constructor, above.

NOTE: Your net will probably require re-training after adding nodes.


=item $net->extend_layer($layer,\%hash);

With this you can modify only one layer with its specifications in a hash refrence. This hash
refrence uses the same keys as for the last new() constructor form, above. 

You can also specify just the number of nodes for the layer in this form:

	$net->extend_layer(0,5);

Which will set the number of nodes in layer 0 to 5 nodes. This is the same as calling:
	
	$net->add_nodes(0,5);

Which does the exact same thing. See add_nodes() below.

NOTE: Your net will probably require re-training after adding nodes.


=item $net->add_nodes($layer,$total_nodes);

This method was created mainly to service the extend*() group of functions, but it 
can also be called independently. This will add nodes as needed to layer C<$layer> to 
make the nodes in layer equal to $total_nodes. 

NOTE: Your net will probably require re-training after adding nodes.



=item $net->p($a,$b);

Returns a floating point number which represents $a as a percentage of $b.



=item $net->intr($float);

Rounds a floating-point number rounded to an integer using sprintf() and int() , Provides
better rounding than just calling int() on the float. Also used very heavily internally.



=item $net->high($array_ref);

Returns the index of the element in array REF passed with the highest comparative value.



=item $net->low($array_ref);

Returns the index of the element in array REF passed with the lowest comparative value.



=item $net->pdiff($array_ref_A, $array_ref_B);

This function is used VERY heavily internally to calculate the difference in percent
between elements of the two array refs passed. It returns a %.20f (sprintf-format) 
percent sting.




=item $net->show();

This will dump a simple listing of all the weights of all the connections of every neuron
in the network to STDIO.




=item $net->crunch($string);

This splits a string passed with /[\s\t]/ into an array ref containing unique indexes
to the words. The words are stored in an intenal array and preserved across load() and save()
calls. This is designed to be used to generate unique maps sutible for passing to learn() and 
run() directly. It returns an array ref.

The words are not duplicated internally. For example:

	$net->crunch("How are you?");

Will probably return an array ref containing 1,2,3. A subsequent call of:

    $net->crunch("How is Jane?");

Will probably return an array ref containing 1,4,5. Notice, the first element stayed
the same. That is because it already stored the word "How". So, each word is stored
only once internally and the returned array ref reflects that.


=item $net->uncrunch($array_ref);

Uncrunches a map (array ref) into an scalar string of words seperated by ' ' and returns the 
string. This is ment to be used as a counterpart to the crunch() method, above, possibly to 
uncrunch() the output of a run() call. Consider the below code (also in ./examples/ex1.pl):
                           
	use AI::NeuralNet::Mesh;
	my $net = AI::NeuralNet::Mesh->new(2,3);
	
	for (0..3) {
		$net->learn_set([
			$net->crunch("I love chips."),  $net->crunch("That's Junk Food!")),
			$net->crunch("I love apples."), $net->crunch("Good, Healthy Food.")),
			$net->crunch("I love pop."),    $net->crunch("That's Junk Food!")),
			$net->crunch("I love oranges."),$net->crunch("Good, Healthy Food."))
		]);
	}
	
	print $net->run_uc("I love corn.")),"\n";


On my system, this responds with, "Good, Healthy Food." If you try to run crunch() with
"I love pop.", though, you will probably get "Food! apples. apples." (At least it returns
that on my system.) As you can see, the associations are not yet perfect, but it can make
for some interesting demos!



=item $net->crunched($word);

This will return undef if the word is not in the internal crunch list, or it will return the
index of the word if it exists in the crunch list. 

If the word is not in the list, it will set the internal error value with a text message
that you can retrive with the error() method, below.

=item $net->word($word);

A function alias for crunched().


=item $net->col_width($width);

This is useful for formating the debugging output of Level 4 if you are learning simple 
bitmaps. This will set the debugger to automatically insert a line break after that many
elements in the map output when dumping the currently run map during a learn loop.

It will return the current width when called with a 0 or undef value.

The column width is preserved across load() and save() calls.


=item $net->random($rand);

This will set the randomness factor from the network. Default is 0.001. When called 
with no arguments, or an undef value, it will return current randomness value. When
called with a 0 value, it will disable randomness in the network. The randomness factor
is preserved across load() and save() calls. 


=item $net->const($const);

This sets the run const. for the network. The run const. is a value that is added
to every input line when a set of inputs are run() or learn() -ed, to prevent the
network from hanging on a 0 value. When called with no arguments, it returns the current
const. value. It defaults to 0.0001 on a newly-created network. The run const. value
is preserved across load() and save() calls.


=item $net->error();

Returns the last error message which occured in the mesh, or undef if no errors have
occured.


=item $net->load_pcx($filename);

NOTE: To use this function, you must have PCX::Loader installed. If you do not have
PCX::Loader installed, it will return undef and store an error for you to retrive with 
the error() method, below.

This is a treat... this routine will load a PCX-format file (yah, I know ... ancient 
format ... but it is the only one I could find specs for to write it in Perl. If 
anyone can get specs for any other formats, or could write a loader for them, I 
would be very grateful!) Anyways, a PCX-format file that is exactly 320x200 with 8 bits 
per pixel, with pure Perl. It returns a blessed refrence to a PCX::Loader object, which 
supports the following routinges/members. See example files ex_pcx.pl and ex_pcxl.pl in 
the ./examples/ directory.

The methods below are basically the same as what you would find when you type:

	% perldoc PCX::Loader



=item $pcx->{image}

This is an array refrence to the entire image. The array containes exactly 64000 elements, each
element contains a number corresponding into an index of the palette array, details below.



=item $pcx->{palette}

This is an array ref to an AoH (array of hashes). Each element has the following three keys:
	
	$pcx->{palette}->[0]->{red};
	$pcx->{palette}->[0]->{green};
	$pcx->{palette}->[0]->{blue};

Each is in the range of 0..63, corresponding to their named color component.



=item $pcx->get_block($array_ref);

Returns a rectangular block defined by an array ref in the form of:
	
	[$left,$top,$right,$bottom]

These must be in the range of 0..319 for $left and $right, and the range of 0..199 for
$top and $bottom. The block is returned as an array ref with horizontal lines in sequental order.
I.e. to get a pixel from [2,5] in the block, and $left-$right was 20, then the element in 
the array ref containing the contents of coordinates [2,5] would be found by [5*20+2] ($y*$width+$x).
    
	print $pcx->get_block(0,0,20,50)->[5*20+2];

This would print the contents of the element at block coords [2,5].



=item $pcx->get($x,$y);

Returns the value of pixel at image coordinates $x,$y.
$x must be in the range of 0..319 and $y must be in the range of 0..199.



=item $pcx->rgb($index);

Returns a 3-element array (not array ref) with each element corresponding to the red, green, or
blue color components, respecitvely.



=item $pcx->avg($index);	

Returns the mean value of the red, green, and blue values at the palette index in $index.

	

=head1 WHAT CAN IT DO?

Rodin Porrata asked on the ai-neuralnet-backprop malining list,
"What can they [Neural Networks] do?". In regards to that questioin,
consider the following:

Neural Nets are formed by simulated neurons connected together much the same
way the brain's neurons are, neural networks are able to associate and
generalize without rules.  They have solved problems in pattern recognition,
robotics, speech processing, financial predicting and signal processing, to
name a few.

One of the first impressive neural networks was NetTalk, which read in ASCII
text and correctly pronounced the words (producing phonemes which drove a
speech chip), even those it had never seen before.  Designed by John Hopkins
biophysicist Terry Sejnowski and Charles Rosenberg of Princeton in 1986,
this application made the Backprogagation training algorithm famous.  Using
the same paradigm, a neural network has been trained to classify sonar
returns from an undersea mine and rock.  This classifier, designed by
Sejnowski and R.  Paul Gorman, performed better than a nearest-neighbor
classifier.

The kinds of problems best solved by neural networks are those that people
are good at such as association, evaluation and pattern recognition.
Problems that are difficult to compute and do not require perfect answers,
just very good answers, are also best done with neural networks.  A quick,
very good response is often more desirable than a more accurate answer which
takes longer to compute.  This is especially true in robotics or industrial
controller applications.  Predictions of behavior and general analysis of
data are also affairs for neural networks.  In the financial arena, consumer
loan analysis and financial forecasting make good applications.  New network
designers are working on weather forecasts by neural networks (Myself
included).  Currently, doctors are developing medical neural networks as an
aid in diagnosis.  Attorneys and insurance companies are also working on
neural networks to help estimate the value of claims.

Neural networks are poor at precise calculations and serial processing. They
are also unable to predict or recognize anything that does not inherently
contain some sort of pattern.  For example, they cannot predict the lottery,
since this is a random process.  It is unlikely that a neural network could
be built which has the capacity to think as well as a person does for two
reasons.  Neural networks are terrible at deduction, or logical thinking and
the human brain is just too complex to completely simulate.  Also, some
problems are too difficult for present technology.  Real vision, for
example, is a long way off.

In short, Neural Networks are poor at precise calculations, but good at
association, evaluation, and pattern recognition.


=head1 EXAMPLES

Included are several example files in the "examples" directory from the
distribution ZIP file.

	ex_dow.pl
	ex_add.pl
	ex_add2.pl
	ex_pcx.pl
	ex_pcx2.pl
	ex_alpha.pl
	ex_bmp.pl
	ex_bmp2.pl
	ex_letters.pl
	ex_pat.pl
	ex_crunch.pl
	ex_synop.pl
	
Each of these includes a short explanation at the top of the file. Each of these
are ment to demonstrate simple, yet practical uses of this module.
	


=head1 OTHER INCLUDED PACKAGES

These packages are not designed to be called directly, they are for internal use. They are
listed here simply for your refrence.

=item AI::NeuralNet::Mesh::node

This is the worker package of the mesh. It implements all the individual nodes of the mesh.

=item AI::NeuralNet::Mesh::cap

This is applied to the input layer of the mesh to prevent the mesh from trying to recursivly
adjust weights out throug the inputs.

=item AI::NeuralNet::Mesh::output

This is simply a data collector package clamped onto the output layer to record the data 
as it comes out of the mesh. 


=head1 BUGS

This is an alpha release of C<AI::NeuralNet::Mesh>, and that holding true, I am sure 
there are probably bugs in here which I just have not found yet. If you find bugs in this module, I would 
appreciate it greatly if you could report them to me at F<E<lt>jdb@wcoil.comE<gt>>,
or, even better, try to patch them yourself and figure out why the bug is being buggy, and
send me the patched code, again at F<E<lt>jdb@wcoil.comE<gt>>. 



=head1 AUTHOR

Josiah Bryan F<E<lt>jdb@wcoil.comE<gt>>

Copyright (c) 2000 Josiah Bryan. All rights reserved. This program is free software; 
you can redistribute it and/or modify it under the same terms as Perl itself.

The C<AI::NeuralNet::Mesh> and related modules are free software. THEY COME WITHOUT WARRANTY OF ANY KIND.

$Id: AI::NeuralNet::Mesh.pm, v0.31 2000/25/12 05:26:10 josiah Exp $


=head1 THANKS

Below are a list of the people that have contributed in some way to this module (no particular order):

	Rodin Porrata, rodin@ursa.llnl.gov
	Randal L. Schwartz, merlyn@stonehedge.com
	Michiel de Roo, michiel@geo.uu.nl
	
Thanks to Randal and Michiel for spoting some documentation and makefile bugs in the last release.
Thanks to Rodin for the prompting to get the size control for individual layers in the network, as well
as asking for negative weights.

=head1 DOWNLOAD

You can always download the latest copy of AI::NeuralNet::Mesh
from http://www.josiah.countystart.com/modules/get.pl?mesh:pod


=head1 MAILING LIST

A mailing list has been setup for AI::NeuralNet::Mesh and AI::NeuralNet::BackProp. 
The list is for discussion of AI and neural net related topics as they pertain to 
AI::NeuralNet::BackProp and AI::NeuralNet::mesh. I will also announce in the group
each time a new release of AI::NeuralNet::Mesh is available.

The list address is at:
	 ai-neuralnet-backprop@egroups.com 
	 
To subscribe, send a blank email:
	ai-neuralnet-backprop-subscribe@egroups.com  


=cut
