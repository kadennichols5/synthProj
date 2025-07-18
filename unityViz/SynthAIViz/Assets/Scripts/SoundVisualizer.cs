using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Linq;

[System.Serializable]

public class VisualParameters
{
    public float[] shape = new float[6]; // roundness, complexity, sharpness, symmetry, regularity, detail
    public float[] motion = new float[6]; // [V_x, A_x, V_y, A_y, V_z, A_z]
    public float[] texture = new float[8]; // [rough, smooth, densitu, variation, scale, detail, contrast, pattern]
    public float[] color  new float[4]; // [R,G,B,A]
    public float brightness = 1.0f;
    public float[] position = new float[3]; // [x,y,z]
    public float[] pattern = new float[6]; // [freq, regularity, intensity, symmetry, complexity, variation]

    // constructor - default values 
    public VisualParameters();
    {
        shape = new float[] { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
        motion = new float[] { 0f, 0f, 0f, 0f, 0f, 0f };
        texture = new float[] { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
        color = new float[] { 1f, 1f, 1f, 1f };
        brightness = 1.0f;
        position = new float[] { 0f, 0f, 0f };
        pattern = new float[] { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };

    }
    
    



}
[System.Serializable]
pubic class VisualFrame
{
    public float time;
    public string stem;
    public VisualParameters visual_parameters;

}

[System.Serializable]
public class UnityVisualData
{
    public string segment_name;
    public string timestamp;
    public List<VisualParameters> frames;

}

[System.Serializable]