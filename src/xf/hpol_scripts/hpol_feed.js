// Creating the feed (for HPOL)
function CreateAntennaSource(zpos_ground, zpos_feed)
{
    // Here we will create our waveform, create our circuit component definition for the feed, and create
    // a CircuitComponent that will attach those to our current geometry.
    var waveformList = App.getActiveProject().getWaveformList();
	waveformList.clear();
    // Create a gaussian derivative input wave
    var waveform = new Waveform();
    var GDer = new GaussianDerivativeWaveformShape ();
    GDer.pulseWidth = 2e-9;
    waveform.setWaveformShape( GDer );
    waveform.name ="Gaussian Derivative";
    var waveformInList = waveformList.addWaveform( waveform );

    // Now to create the circuit component definition:
    var componentDefinitionList = App.getActiveProject().getCircuitComponentDefinitionList();
    // clear the list
    componentDefinitionList.clear();
    // Create our Feed
    var feed = new Feed();
    feed.feedType = Feed.Voltage; // Set its type enumeration to be Voltage.
    // Define a 50-ohm resistance for this feed
    var rlc = new RLCSpecification();
    rlc.setResistance( "50 ohm" );
    rlc.setCapacitance( "0" );
    rlc.setInductance( "0" );
    feed.setImpedanceSpecification( rlc );
    feed.setWaveform( waveformInList );  // Make sure to use the reference that was returned by the list, or query the list directly
    feed.name = "50-Ohm Voltage Source";
    var feedInList = componentDefinitionList.addCircuitComponentDefinition( feed );

    // Now create a circuit component that will be the feed point for our simulation
    var componentList = App.getActiveProject().getCircuitComponentList();
    componentList.clear();

    var component = new CircuitComponent();
    component.name = "Bottom Feed";
    component.setAsPort( true );
    // Define the endpoints of this feed - these are defined in world position, but you can also attach them to edges, faces, etc.
    var coordinate1 = new CoordinateSystemPosition( 0 + units, 0 + units, zpos_ground + units);
    var coordinate2 = new CoordinateSystemPosition( 0 + units, 0 + units, (zpos_feed - (feed_dist/2)) + units);
    component.setCircuitComponentDefinition( feedInList );
    component.setEndpoint1( coordinate1 );
    component.setEndpoint2( coordinate2 );
    componentList.addCircuitComponent( component );

    // Now we need to add the capacitor to the top half of the Feed
    var cap = new PassiveLoad();
    cap.feedType = Feed.Voltage; // Set its type enumeration to be Voltage.
    // Define a capacitance for this feed 
    var rlc2 = new RLCSpecification();
    rlc2.setResistance( "0" );
    rlc2.setCapacitance( "20 pF" );
    rlc2.setInductance( "0" );
    cap.setImpedanceSpecification( rlc2 );
    cap.name = "20 pF Capacitor";
    var capInList = componentDefinitionList.addCircuitComponentDefinition( cap );

    var capacitor = new CircuitComponent();
    capacitor.name = "Capacitor";
    capacitor.setAsPort( true );
    // Define the endpoints of this capacitor - these are defined in world position, but you can also attach them to edges, faces, etc.
    var coordinate3 = new CoordinateSystemPosition( 0 + units, 0 + units, (zpos_ground + (feed_dist/2)) + units);
    var coordinate4 = new CoordinateSystemPosition( 0 + units, 0 + units, zpos_feed + units);
    capacitor.setCircuitComponentDefinition( capInList );
    capacitor.setEndpoint1( coordinate3 );
    capacitor.setEndpoint2( coordinate4 );
    componentList.addCircuitComponent( capacitor );
}
