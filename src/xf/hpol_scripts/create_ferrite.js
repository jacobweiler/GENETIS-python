// ferrite function defintion
function CreateFerrite()
{
    // Adding in sampled data (based on stephs fs-20 material)
    var mSampled = new MagneticSampledParameters()
    mSampled.addFrequency("0.1 GHz",15, "413 ohm/m")
    mSampled.addFrequency("0.2 GHz",18, "1990 ohm/m")
    mSampled.addFrequency("0.3 GHz",16, "26530 ohm/m")
    mSampled.addFrequency("0.4 GHz",11, "34741 ohm/m")
    mSampled.addFrequency("0.5 GHz",10, "39500 ohm/m")
    mSampled.addFrequency("0.6 GHz",10, "47400 ohm/m")
    mSampled.addFrequency("0.7 GHz",10, "55300 ohm/m")
    mSampled.addFrequency("0.8 GHz",10, "63200 ohm/m")
    mSampled.addFrequency("0.9 GHz",10, "71100 ohm/m")
    mSampled.addFrequency("1 GHz",10, "79000 ohm/m")

    var ferrite = new Material();
    ferrite.name = "ferrite";
    var ferriteElectricIsotropic = new ElectricIsotropic();

    var electricNormalParams = new ElectricNormalParameters();
    electricNormalParams.setConductivity("0 S/m");
    electricNormalParams.setRelativePermittivity("10");
    ferriteElectricIsotropic.setParameters(electricNormalParams);

    var ferriteMagneticIsotropic = new MagneticIsotropic(mSampled);
    var ferritePhysicalMaterial = new PhysicalMaterial();
    ferritePhysicalMaterial.setElectricProperties( ferriteElectricIsotropic );
    ferritePhysicalMaterial.setMagneticProperties( ferriteMagneticIsotropic );

    ferrite.setDetails( ferritePhysicalMaterial );

    var ferriteBodyAppearance = ferrite.getAppearance();
    var ferriteFaceAppearance = ferriteBodyAppearance.getFaceAppearance();
    ferriteFaceAppearance.setColor( new Color( 200, 100, 200, 200) );

    if(null != App.getActiveProject().getMaterialList().getMaterial(ferrite.name) )
    {
        App.getActiveProject().getMaterialList().removeMaterial(ferrite.name);
    }

    App.getActiveProject().getMaterialList().addMaterial( ferrite );

}
