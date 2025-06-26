var start = gen * popsize + 1;
var end = start + popsize - 1;

for (var k = start; k <= end; k++){
    // if var simNum doesn't exist, create it
    var simNum = k;
    var simlength = k.toString().length;
    
    var totchars = 6;
    var init = "";
    for (var i = 0; i < totchars - simlength; i++){
        init = init + "0";
    }
    var simNum = init + simNum;
    Output.println("simNum is");
    Output.println(simNum);
    var query = new ResultQuery();
    ///////////////////////Get Theta and Phi Gain///////////////
    query.projectId = App.getActiveProject().getProjectDirectory();
    Output.println( App.getActiveProject().getProjectDirectory() );

    query.runId = "Run0001";
    query.simulationId = simNum;
    query.sensorType = ResultQuery.FarZoneSensor;
    query.sensorId = "Far Zone Sensor";
    query.timeDependence = ResultQuery.SteadyState;
    query.resultType = ResultQuery.RealizedGain; // Toggle ideal + realized gain here
    query.fieldScatter = ResultQuery.TotalField;
    query.resultComponent = ResultQuery.Theta;
    query.dataTransform = ResultQuery.NoTransform;
    query.complexPart = ResultQuery.NotComplex;
    query.surfaceInterpolationResolution = ResultQuery.NoInterpolation;
    query.setDimensionRange( "Frequency", 0, -1 );
    query.setDimensionRange( "Theta", 0, -1 );
    query.setDimensionRange( "Phi", 0, -1 );

    var thdata = new ResultDataSet( "" );
    thdata.setQuery( query );
    if( !thdata.isValid() ){
        Output.println( "1getCurrentDataSet() : " +
        thdata.getReasonWhyInvalid() );
    }

    query.resultComponent = ResultQuery.Phi;
    var phdata = new ResultDataSet("");
    phdata.setQuery(query);

    if( !phdata.isValid() ){
        Output.println( "2getCurrentDataSet() : " +
        phdata.getReasonWhyInvalid() );
    }
    /////////////////Get Theta and Phi Phase///////////////////////////////////

    query.resultType = ResultQuery.E;
    query.fieldScatter = ResultQuery.TotalField;
    query.resultComponent = ResultQuery.Theta;
    query.dataTransform = ResultQuery.NoTransform;
    query.complexPart = ResultQuery.Phase;
    query.surfaceInterpolationResolution = ResultQuery.NoInterpolation;
    query.setDimensionRange( "Frequency", 0, -1 );
    query.setDimensionRange( "Theta", 0, -1 );
    query.setDimensionRange( "Phi", 0, -1 );


    var thphase = new ResultDataSet("");
    thphase.setQuery(query);

    if( !thphase.isValid() ){
        Output.println( "3getCurrentDataSet() : " +
        thphase.getReasonWhyInvalid() );
    }

    query.resultComponent = ResultQuery.Phi;
    query.ComplexPart = ResultQuery.Phase;
    var phphase = new ResultDataSet("");
    phphase.setQuery(query);

    if( !phphase.isValid() ){
        Output.println( "4getCurrentDataSet() : " +
        phphase.getReasonWhyInvalid() );
    }

    /////////////////Get Input Power///////////////////////////
    query.sensorType = ResultQuery.System;
    query.sensorId = "System";
    query.timeDependence = ResultQuery.SteadyState;
    query.resultType = ResultQuery.NetInputPower;
    query.fieldScatter = ResultQuery.NoFieldScatter;
    query.resultComponent = ResultQuery.Scalar;
    query.dataTransform = ResultQuery.NoTransform;
    query.complexPart = ResultQuery.NotComplex;
    query.surfaceInterpolationResolution = ResultQuery.NoInterpolation;
    query.clearDimensions();
    query.setDimensionRange("Frequency",0,-1);

    var inputpower = new ResultDataSet("");
    inputpower.setQuery(query);
    if( !inputpower.isValid() ){
        Output.println( "5getCurrentDataSet() : " +
        inputpower.getReasonWhyInvalid() );
    }
    var freqCoefficients = 60;
    for (var i = 1; i <= freqCoefficients; i++){
        if (typeof node === 'undefined'){
            var ind_num = k - (gen * popsize);
            var file = RunDir + "/Generation_Data/" + gen + "/uan_files/"+ ind_num + "/";
            file = file + gen + "_" + ind_num + "_";
            file = file + (i) + ".uan";
        } else {
            var file = RunDir + "/tempFiles/" + node + "/uan_files/";
            file = file + "1_1_" + (i) + ".uan";
        }
        Output.println(file);
        Output.println("thdata: " + thdata );
        Output.println("thphase: " + thphase);
        Output.println("phdata: " + phdata);
        Output.println("phphase: " + phphase);
        Output.println("inputpower: " + inputpower);
        Output.println("i: " + i);
        FarZoneUtils.exportToUANFile(thdata,thphase,phdata,phphase,inputpower,file, i-1);

        // VSWR, S11, Impedance
        query.sensorType = ResultQuery.CircuitComponent;
        query.sensorId = "Source";
        query.timeDependence = ResultQuery.SteadyState;
        query.resultType = ResultQuery.VSWR;
        query.fieldScatter = ResultQuery.NoFieldScatter;
        query.resultComponent = ResultQuery.Scalar;
        query.dataTransform = ResultQuery.NoTransform;
        query.complexPart = ResultQuery.NotComplex;
        query.surfaceInterpolationResolution = ResultQuery.NoInterpolation;
        query.setDimensionRange( "Frequency", 0, -1 );
        var vswr = new ResultDataSet("");
        vswr.setQuery(query);

        if( !vswr.isValid() ){
            Output.println( "6getCurrentDataSet() : " +
            vswr.getReasonWhyInvalid() );
        }

        query.sensorType = ResultQuery.CircuitComponent;
        query.sensorId = "Source";
        query.timeDependence = ResultQuery.SteadyState;
        query.resultType = ResultQuery.ReflectionCoefficient;
        query.fieldScatter = ResultQuery.NoFieldScatter;
        query.resultComponent = ResultQuery.Scalar;
        query.dataTransform = ResultQuery.NoTransform;
        query.complexPart = ResultQuery.ComplexMagnitude;
        query.surfaceInterpolationResolution = ResultQuery.NoInterpolation;
        query.setDimensionRange( "Frequency", 0, -1 );
        var s11 = new ResultDataSet("");
        s11.setQuery(query);

        if( !s11.isValid() ){
            Output.println( "7getCurrentDataSet() : " +
            s11.getReasonWhyInvalid() );
        }
        /*
        query.sensorType = ResultQuery.CircuitComponent;
        query.sensorId = "Source";
        query.timeDependence = ResultQuery.SteadyState;
        query.resultType = ResultQuery.Impedance;
        query.fieldScatter = ResultQuery.NoFieldScatter;
        query.resultComponent = ResultQuery.Scalar;
        query.dataTransform = ResultQuery.NoTransform;
        query.complexPart = ResultQuery.NotComplex;
        query.surfaceInterpolationResolution = ResultQuery.NoInterpolation;
        query.setDimensionRange( "Frequency", 0, -1 );
        var imp = new ResultDataSet("");
        imp.setQuery(query);

        if( !imp.isValid() ){
            Output.println( "8getCurrentDataSet() : " +
            imp.getReasonWhyInvalid() );
        }
        */
        var ind_num = k - (gen * popsize);
        var file = RunDir + "/Generation_Data/" + gen + "/";
        file = file + gen + "_" + ind_num;
        file = file + "_vswr_s11_imp.csv";

        Output.println(file);

        var file_to_open = new File( file );
        file_to_open.open(IODevice.WriteOnly); 

        //DataSetExportUtility.exportDataSetCsv(file_to_open, [vswr, s11, imp], false);
        DataSetExportUtility.exportDataSetCsv(file_to_open, [vswr, s11], false);
        file_to_open.close();
    }
}
App.quit();
