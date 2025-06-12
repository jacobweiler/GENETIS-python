/********************************************* Function Calls **************************************************/
// Lists to hold the genes
if (CURVED == 0) { // Straight Sides
    if (NSECTIONS == 0) {
        Output.println("Straight Sided Asym Antenna")
        var radii1=[]; 
        var lengths1=[]; 
        var angles1=[]; 
        var sep=[];
        var radii2=[]; 
        var lengths2=[];
        var angles2=[];
    }
    else {
        Output.println("Straight Sided Sym Antenna")
        var radii=[];
        var lengths=[];
        var angles=[];
    }
}
else { // Curved 
    Output.println("Curved Asym Antenna")
    var radii1=[];
    var lengths1=[];
    var A1=[];
    var B1=[];
    var radii2=[];
    var lengths2=[];
    var A2=[];
    var B2=[];
}

var lines = generationDNA.split('\n');

// Loop over reading in the gene values
for(var i = 0;i < lines.length - 1;i++){
	if(i>=antennaLines) {
        // Split data and then read out individual
        var params = lines[i].split(",");
		Output.println("Individual "+ i);
        if (CURVED == 0) { // Straight Sides
            if (NSECTIONS == 0) {
                Output.println("Straight Sided Asym Antenna")

                Output.println("radii1: "+params[0]);
                Output.println("lengths1: "+params[1]);
                Output.println("angles1: "+params[2]);
                Output.println("sep: "+params[3]);
                Output.println("radii2: "+params[4]);
                Output.println("lengths2: "+params[5]);
                Output.println("angles2: "+params[6]);

                radii1[i-antennaLines]=params[0];
                lengths1[i-antennaLines]=params[1];
                angles1[i-antennaLines]=params[2];
                sep[i-antennaLines]=params[3];
                radii2[i-antennaLines]=params[4];
                lengths2[i-antennaLines]=params[5];
                angles2[i-antennaLines]=params[6];
            }
            else {
                Output.println("Straight Sided Sym Antenna")

                Output.println("radii: "+params[0]);
                Output.println("lengths: "+params[1]);
                Output.println("angles: "+params[2]);

                radii[i-antennaLines]=params[0];
                lengths[i-antennaLines]=params[1];
                angles[i-antennaLines]=params[2];
            }
        }
        else { // Curved 
            Output.println("Curved Asym Antenna")

            Output.println("radii1: "+params[0]);
            Output.println("lengths1: "+params[1]);
            Output.println("A1: "+params[2]);
            Output.println("B1: "+params[3])
            Output.println("radii2: "+params[4]);
            Output.println("lengths2: "+params[5]);
            Output.println("A2: "+params[6]);
            Output.println("B2: "+params[7]);

            radii1[i-antennaLines]=params[0];
            lengths1[i-antennaLines]=params[1];
            A1[i-antennaLines]=params[2];
            B1[i-antennaLines]=params[3];
            radii2[i-antennaLines]=params[4];
            lengths2[i-antennaLines]=params[5];
            A2[i-antennaLines]=params[6];
            B2[i-antennaLines]=params[7];
        }
	}
}

for(var i = indiv - 1; i < NPOP; i++) {
    App.getActiveProject().getGeometryAssembly().clear();
    CreatePEC();
    // Calling the same function with different inputted variables depending on the version of the loop we are in
    if (CURVED == 0) { // Straight Sides
        if (NSECTIONS == 0) {
            Output.println("Straight Sided Asym Antenna")
            var radius1 = radii1[i]; 
            var length1 = lengths1[i]; 
            var angle1 = angles1[i]; 
            var seperation = sep[i];
            var radius2 = radii2[i]; 
            var length2 = lengths2[i];
            var angle2 = angles2[i];
            Output.println('radius1: ' + radius1);
            Output.println('length1: ' + length1);
            Output.println('angle1: ' + angle1);
            Output.println('seperation: ' + seperation);
            Output.println('radius2: ' + radius2);
            Output.println('length2: ' + length2);
            Output.println('angle2: ' + angle2);
            if (evolve_sep == 1){ // evolving seperation! (might need to change the min max values in GA Scripts)
                var sepDist = seperation;
            }
            build_vpol(radius1, length1, angle1, radius2, length2, angle2);
        }
        else {
            Output.println("Straight Sided Sym Antenna")
            var radius = radii[i];
            var length = lengths[i];
            var angle = angles[i];
            Output.println('radius: ' + radius);
            Output.println('length: ' + length);
            Output.println('angle: ' + angle)

            build_vpol(radius, length, angle);
        }
    }
    else { // Curved 
        Output.println("Curved Asym Antenna")
        var radius1 = radii1[i];
        var length1 = lengths1[i];
        var a_1 = A1[i];
        var b_1 = B1[i];
        var radius2 = radii2[i];
        var length2 = lengths2[i];
        var a_2 = A2[i];
        var b_2 = B2[i];
        Output.println('radius1: ' + radius1);
        Output.println('length1: ' + length1);
        Output.println('a_1: ' + a_1);
        Output.println('b_1: ' + b_1);
        Output.println('radius2: ' + radius2);
        Output.println('length2: ' + length2);
        Output.println('a_2: ' + a_2);
        Output.println('b_2: ' + b_2);

        build_vpol(radius1, length1, false, radius2, length2, false, a_1, b_1, a_2, b_2);
    }
    CreateAntennaSource(); 
    CreateGrid();
    CreateSensors();
    CreateAntennaSimulationData();
    QueueSimulation();
    Output.println(ResultQuery().simulationId);
    MakeImage(i);
}

file.close();
App.quit();
