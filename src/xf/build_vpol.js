// vpol building scripts
function build_vpol(radius1, length1, angle1, radius2, length2, angle2, a_1, b_1, a_2, b_2) 
{
    // Building scripts for different versions of the loop :) 
    if (CURVED == 0){ // Straight Sides
        if (NSECTIONS == 0){
            Output.println("Straight Sided Asym Antenna Building")

            //create a new sketch - this is the base of the antenna
            var segment1 = new Sketch();
            var segment2 = new Sketch();
            var base1 = new Ellipse( new Cartesian3D(0,0,0), new Cartesian3D( radius1 + units,0,0 ), 1.0, 0.0, Math.PI*2 );
            var base2 = new Ellipse( new Cartesian3D(0,0,0), new Cartesian3D( radius2 + units,0,0 ), 1.0, 0.0, Math.PI*2 );
            segment1.addEdge(base1);
            segment2.addEdge(base2);

            //extrude function with a draft - this extends the antenna outward, with its edges shifted at angle defined by DraftAngle
            var extrudeLeft = new Extrude( segment1, length1 + units );
            var newOptionsLeft = extrudeLeft.getOptions();
            newOptionsLeft.draftAngle = angle1;
            newOptionsLeft.draftOption = 1;
            extrudeLeft.setOptions ( newOptionsLeft );

            //create a recipe and model
            var segmentLeftRecipe = new Recipe();
            segmentLeftRecipe.append(extrudeLeft);
            var segmentLeftModel = new Model();
            segmentLeftModel.setRecipe(segmentLeftRecipe);
            segmentLeftModel.name = "Antenna Test Segment - Left " + (i+1);

            //add depth to the circle - right side
            extrudeRightDirection = CoordinateSystemDirection(0,0,-1);
            var extrudeRight = new Extrude( segment2, length2 + units, extrudeRightDirection );
            var newOptionsRight = extrudeRight.getOptions();
            newOptionsRight.draftAngle = angle2;
            newOptionsRight.draftOption = 1;
            extrudeRight.setOptions ( newOptionsRight );

            //create a recipe and model
            var segmentRightRecipe = new Recipe();
            segmentRightRecipe.append(extrudeRight);
            var segmentRightModel = new Model();
            segmentRightModel.setRecipe(segmentRightRecipe);
            segmentRightModel.name = "Antenna Test Segment - Right " + (i+1);

            //set locations of the left and right segments
            segmentLeftModel.getCoordinateSystem().translate(new Cartesian3D(0,0,sepDist + units));
            var segmentInProject1 = App.getActiveProject().getGeometryAssembly().append(segmentLeftModel);
            segmentRightModel.getCoordinateSystem().translate(new Cartesian3D(0,0,0));
            var segmentInProject2 = App.getActiveProject().getGeometryAssembly().append(segmentRightModel);

            // Now set the material for the Antenna:
            var pecMaterial = App.getActiveProject().getMaterialList().getMaterial( "PEC" );
            if( null == pecMaterial ){
                Output.println( "\"PEC\" material was not found, could not associate with the antenna." );
            }
            else{
                App.getActiveProject().setMaterial( segmentInProject1, pecMaterial );
                App.getActiveProject().setMaterial( segmentInProject2, pecMaterial );
            }

            //zoom to view the extent of the creation
            View.zoomToExtents();
        }
        else {
            Output.println("Straight Sided Sym Antenna")

            //create a new sketch - this is the base of the antenna
            var segment = new Sketch();
            var base = new Ellipse( new Cartesian3D(0,0,0), new Cartesian3D( radius1 + units,0,0 ), 1.0, 0.0, Math.PI*2 );
            segment.addEdge(base);

            //extrude function with a draft - this extends the antenna outward, with its edges shifted at angle defined by DraftAngle
            var extrudeLeft = new Extrude( segment, length1 + units );
            var newOptionsLeft = extrudeLeft.getOptions();
            newOptionsLeft.draftAngle = angle1;
            newOptionsLeft.draftOption = 1;
            extrudeLeft.setOptions ( newOptionsLeft );

            //create a recipe and model
            var segmentLeftRecipe = new Recipe();
            segmentLeftRecipe.append(extrudeLeft);
            var segmentLeftModel = new Model();
            segmentLeftModel.setRecipe(segmentLeftRecipe);
            segmentLeftModel.name = "Antenna Test Segment - Left " + (i+1);

            //add depth to the circle - right side
            extrudeRightDirection = CoordinateSystemDirection(0,0,-1);
            var extrudeRight = new Extrude( segment, length1 + units, extrudeRightDirection );
            var newOptionsRight = extrudeLeft.getOptions();
            newOptionsRight.draftAngle = angle1;
            newOptionsRight.draftOption = 1;
            extrudeRight.setOptions ( newOptionsRight );

            //create a recipe and model
            var segmentRightRecipe = new Recipe();
            segmentRightRecipe.append(extrudeRight);
            var segmentRightModel = new Model();
            segmentRightModel.setRecipe(segmentRightRecipe);
            segmentRightModel.name = "Antenna Test Segment - Right " + (i+1);

            //set locations of the left and right segments
            segmentLeftModel.getCoordinateSystem().translate( new Cartesian3D(0,0,sepDist + units)); 
            var segmentInProject1 = App.getActiveProject().getGeometryAssembly().append(segmentLeftModel);
            segmentRightModel.getCoordinateSystem().translate(new Cartesian3D(0,0,0));
            var segmentInProject2 = App.getActiveProject().getGeometryAssembly().append(segmentRightModel);

            // Now set the material for the Antenna:
            var pecMaterial = App.getActiveProject().getMaterialList().getMaterial( "PEC" );
            if( null == pecMaterial ) {
                Output.println( "\"PEC\" material was not found, could not associate with the antenna." );
            }
            else {
                App.getActiveProject().setMaterial( segmentInProject1, pecMaterial );
                App.getActiveProject().setMaterial( segmentInProject2, pecMaterial );
            }

            //zoom to view the extent of the creation
            View.zoomToExtents();
        }
    }
    else { // Curved 
        Output.println("Curved Asym Antenna")

        //create a new sketch - this is the base of the antenna
        var segment1 = new Sketch();
        var segment2 = new Sketch();
        var base1 = new Ellipse( new Cartesian3D(0,0,0), new Cartesian3D( radius1 + units,0,0 ), 1.0, 0.0, Math.PI*2 );
        var base2 = new Ellipse( new Cartesian3D(0,0,0), new Cartesian3D( radius2 + units,0,0 ), 1.0, 0.0, Math.PI*2 );
        segment1.addEdge(base1);
        segment2.addEdge(base2);

        //extrude function with a draft - 
        var extrudeLeft = new Extrude( segment1, length1 + units );
        var newOptionsLeft = extrudeLeft.getOptions();
        newOptionsLeft.draftLaw = "10*("+a_1+"*(x/10)^2+"+b_1+"*(x/10))";
        newOptionsLeft.draftOption = 4;
        extrudeLeft.setOptions ( newOptionsLeft );

        //create a recipe and model
        var segmentLeftRecipe = new Recipe();
        segmentLeftRecipe.append(extrudeLeft);
        var segmentLeftModel = new Model();
        segmentLeftModel.setRecipe(segmentLeftRecipe);
        segmentLeftModel.name = "Antenna Test Segment - Left " + (i+1);

        //add depth to the circle - right side
        extrudeRightDirection = CoordinateSystemDirection(0,0,-1);
        var extrudeRight = new Extrude( segment2, length2 + units, extrudeRightDirection );
        var newOptionsRight = extrudeRight.getOptions();
        newOptionsRight.draftLaw = "10*("+a_2+"*(x/10)^2+"+b_2+"*(x/10))";
        newOptionsRight.draftOption = 4;
        extrudeRight.setOptions ( newOptionsRight );

        //create a recipe and model
        var segmentRightRecipe = new Recipe();
        segmentRightRecipe.append(extrudeRight);
        var segmentRightModel = new Model();
        segmentRightModel.setRecipe(segmentRightRecipe);
        segmentRightModel.name = "Antenna Test Segment - Right " + (i+1);

        //set locations of the left and right segments
        segmentLeftModel.getCoordinateSystem().translate(new Cartesian3D(0,0,sepDist + units));
        var segmentInProject1 = App.getActiveProject().getGeometryAssembly().append(segmentLeftModel);
        segmentRightModel.getCoordinateSystem().translate(new Cartesian3D(0,0,0));
        var segmentInProject2 = App.getActiveProject().getGeometryAssembly().append(segmentRightModel);

        // Now set the material for the Antenna:
        var pecMaterial = App.getActiveProject().getMaterialList().getMaterial( "PEC" );
        if( null == pecMaterial ) {
            Output.println( "\"PEC\" material was not found, could not associate with the antenna." );
        }
        else { 
            App.getActiveProject().setMaterial( segmentInProject1, pecMaterial );
            App.getActiveProject().setMaterial( segmentInProject2, pecMaterial );
        }

        //zoom to view the extent of the creation
        View.zoomToExtents();
    }
}
