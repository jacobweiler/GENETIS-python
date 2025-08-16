// Gridding Definition (resolution of simulation)
function CreateGrid()
{
    // Set up the grid spacing for the antenna
    var grid = App.getActiveProject().getGrid();
    var cellSizes = grid.getCellSizesSpecification();
    useProGridCellSizes = false;
    cellSizes.setTargetSizes( Cartesian3D( 0.4 + units, 0.4 + units, 0.4 + units ) );
    // And we need to set the Minimum Sizes - these are the minimum deltas that we will allow in this project.
    // We'll use the scalar ratio of 20% here.
    cellSizes.setMinimumSizes( Cartesian3D( ".05", ".05", ".05" ) );
    cellSizes.setMinimumIsRatioX( true );
    cellSizes.setMinimumIsRatioY( true );
    cellSizes.setMinimumIsRatioZ( true );

    grid.specifyPaddingExtent( Cartesian3D( "20", "20", "20" ), Cartesian3D( "20", "20", "20" ), true, true );
}
