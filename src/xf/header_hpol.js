/**************************************** Set Global Variables **************************************************/
var units = " cm";
var plate_num = 4; // 4 for now until we implement a generic implementation
var thick = 0.02; // Thickness of the HPOL plates
var feed_dist = .55; // distance between ground and feed
var path = workingdir + "/Run_Outputs/" + RunName + "/Generation_Data/" + gen + "_generationDNA.csv"

// create the frequency array
var freq = [];
for (var i = 0; i < freqCoefficients; i++) {
    freq.push(freq_start + i * freq_step);
}

var antennaLines = 0 // This is how many lines come before the antenna data
var file = new File(path);
file.open(1);
var generationDNA = file.readAll();
