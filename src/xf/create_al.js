function CreateAl()
{
    var aluminum = new Material();
    aluminum.name = "aluminum";
    var aluminumElectricIsotropic = new ElectricIsotropic();

    //WALTER - Define electric normal properties and assign them to the electric isotropic object
    var electricNormalParams = new ElectricNormalParameters();
    electricNormalParams.setConductivity("3.7e+07 S/m");
    electricNormalParams.setRelativePermittivity("1");
    aluminumElectricIsotropic.setParameters(electricNormalParams);

    //WALTER - Based on your screenshot, the magnetic properties should be set to free space
    //That is the default for material properties we do not create, so we can ignore the magnetic properties.
    //var aluminumMagneticIsotropic = new MagneticIsotropic();

    var aluminumPhysicalMaterial = new PhysicalMaterial();
    aluminumPhysicalMaterial.setElectricProperties( aluminumElectricIsotropic );
    //aluminumPhysicalMaterial.setMagneticProperties( aluminumMagneticIsotropic );
    aluminum.setDetails( aluminumPhysicalMaterial );

    var aluminumBodyAppearance = aluminum.getAppearance();
    var aluminumFaceAppearance = aluminumBodyAppearance.getFaceAppearance();
    aluminumFaceAppearance.setColor( new Color( 192, 192, 240, 255) );

    if(null != App.getActiveProject().getMaterialList().getMaterial(aluminum.name) )
    {
        App.getActiveProject().getMaterialList().removeMaterial(aluminum.name);
    }
    App.getActiveProject().getMaterialList().addMaterial( aluminum );
}
