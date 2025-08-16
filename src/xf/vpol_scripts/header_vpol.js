/**************************************** Set Global Variables **************************************************/
var units = " cm";
var sepDist = 3.0;
var path = workingdir + "/Run_Outputs/" + RunName + "/Generation_Data/" + gen + "/" + gen + "_generationDNA.csv"
Output.println(path);
// create the frequency array
var freq = [];
for (var i = 0; i < freqCoefficients; i++) {
    freq.push(freq_start + i * freq_step);
}

var antennaLines = 0 // This is how many lines come before the antenna data
var file = new File(path);
file.open(1);
var generationDNA = file.readAll();
