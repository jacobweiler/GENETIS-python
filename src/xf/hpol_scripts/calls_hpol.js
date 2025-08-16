/********************************************* Function Calls **************************************************/
// Lists to hold the genes
var num_plates =[]; // Number of plates
var radius = []; // Radius of the antenna
var arclength = []; // Arc length of the plates
var antenna_height = []; // Height of the antenna
var rod_height = []; // height of the ferrite rods
var rod_radius = []; // radius of the ferrite rods

var lines = generationDNA.split('\n');

// Loop over reading in the gene values
for(var i = 0;i < lines.length - 1;i++){
	if(i>=antennaLines){
		var params = lines[i].split(",");
		Output.println("Individual "+ i);
		Output.println("Number of Plates: "+params[0]);
		Output.println("Radius: "+params[1]);
		Output.println("Arclength: "+params[2]);
		Output.println("Antenna Height: "+params[3]);
        Output.println("Rod Height: "+params[4]);
        Output.println("Rod Radius: "+params[5]);

		num_plates[i-antennaLines]=params[0];
		radius[i-antennaLines]=params[1];
		arclength[i-antennaLines]=params[2];
		antenna_height[i-antennaLines]=params[3];
        rod_height[i-antennaLines]=params[4];
        rod_radius[i-antennaLines]=params[5];
        
	}
}

for(var i = indiv - 1; i < NPOP; i++)
{
    var plate_num = num_plates[i];
    var rad = radius[i];
    var arc = arclength[i];
    var height = antenna_height[i];
    var rod_h = rod_height[i];
    var rod_r = rod_radius[i];

    // Hardcoding in the dimensions that are similar to in-ice HPOL
    var feed_dist = 0.55; // distance between ground and feed
    var rad = 6.5; // Radius of the antenna
    var thick = 0.15; // Thickness of the HPOL plates
    var arc = 2.5; // Arc length of the plates
    var height = 34; // Height of the antenna
    var rod_h = 46.2; // height of the ferrite rods
    var rod_r = 1.85; // radius of the ferrite rods
    var plate_num = 4; // Number of plates

    Output.println('plate_num: ' + plate_num);
    Output.println('rad: ' + rad);
    Output.println('arc: ' + arc);
    Output.println('height: ' + height);
    Output.println('rod_h: ' + rod_h);
    Output.println('rod_r: ' + rod_r);

    App.getActiveProject().getGeometryAssembly().clear();
    CreatePEC();
    CreateFerrite();
    CreateAl();
    build_hpol(plate_num, rad, thick, arc, height, rod_h, rod_r);
    CreateAntennaSource((height/2) - feed_dist, (height/2) + feed_dist); 
    CreateGrid();
    CreateSensors();
    CreateAntennaSimulationData();
    QueueSimulation();
    Output.println(ResultQuery().simulationId);
    MakeImage(i);
}

file.close();
App.quit();
