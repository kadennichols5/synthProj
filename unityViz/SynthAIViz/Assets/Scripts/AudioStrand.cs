
using UnityEngine;
using System.Collections.Generic;

public class AudioStrand : MonoBehaviour
{
    [Header("Audio Strand Properties")]
    public string StemType {get; private set;}
    public Color BaseColor {get; private set;}

    [Header("Visual Components")]
    public LineRenderer lineRenderer;
    public ParticleSystem particles;
    public TrailRenderer trailRenderer;

    [Header("Strand Behavior")]
    public AnimationCurve thicknessCurve = AnimationCurve.Linear(0, 0.1f, 1, 0.5f);
    public AnimationCurve brightnessCurve = AnimationCurve.Linear(0, 0.5f, 1, 1.5f);
    public float maxStrandLength = 50f;
    public int maxStrandPoints = 100;

    // internal state of each audio strand 
    private List<Vector3> strandPoints;
    private List<float> pointTimes;
    private List<VisualParameters> pointParameters;
    private Renderer meshRenderer;
    private MaterialPropertyBlock propertyBlock;
    private Vector3 currentVelocity;
    private Vector3 currentPosition;

    // Fractal and Shape generation parameters
    private Mesh currentMesh;
    private Vector3[] baseVertices;
    private int[] baseTriangles;

    void Awake(){
        InitializeComponents(); 
    }

    void InitializeComponents()
    {
        
        // initialize data structures
        strandPoints = new List<Vector3>();
        pointTimes = new List<float>();
        pointParameters = new List<VisualParameters>();
        propertyBlock = new MaterialPropertyBlock();

        // get or create components
        meshRenderer = GetComponent<Renderer>();
        if (meshRenderer == null)
        {
            meshRenderer = gameObject.AddComponent<MeshRenderer>();
        }
        // create line renderer for strand visualization
        if (lineRenderer == null)
        {
            lineRenderer = gameObject.AddComponent<LineRenderer>();
            SetupLineRenderer();
        }
        // create particle system for texture effects
        if (particles = null)
        {
            GameObject particlesObj = new GameObject("StrandParticles"); //initialize new object
            particleObj.transform.SetParent(transform); // makes particlesObj child of audio strand
            particles = particleObj.AddComponent<ParticleSystem>(); 
            //.adds component data type= ParticleSystem to 
            // game object called particleObj now known as particles
            SetUpParticleSystem();
        }

        if (trailRenderer = null)
        {
            trailRenderer = gameObject.AddComponent<TrailRenderer>();
            SetupTrailRenderer();
        }

        CreateBaseMesh()
    }

    void SetUpLineRenderer()
    {
        lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.startWidth = 0.1f;
        lineRenderer.endWidth = 0.05f;
        lineRenderer.positionCount = 0;
        lineRenderer.useWorldSpace = true;
        lineRenderer.generateLightingData = true;
    }

    void SetUpParticleSystem()
    {
        var main = particles.main;
        main.startLifetime = 2.0f;
        main.startSpeed = 1.0f;
        main.startSize = 0.1f;
        main.startColor = Color.white;
        main.maxParticles = 100;
        
        var emission = particles.emission;
        emission.rateOverTime = 20;
        
        var shape = particles.shape;
        shape.enabled = true;
        shape.shapeType = ParticleSystemShapeType.Sphere;
        shape.radius = 0.5f;
        
        var velocityOverLifetime = particles.velocityOverLifetime;
        velocityOverLifetime.enabled = true;
        velocityOverLifetime.space = ParticleSystemSimulationSpace.Local;
    }

    void SetupTrailRenderer()
    {
        trailRenderer.material = new Material(Shader.Find("Sprites/Default"));
        trailRenderer.time = 1.0f;
        trailRenderer.startWidth = 0.2f;
        trailRenderer.endWidth = 0.01f;
        trailRenderer.generateLightingData = true;
    }
        

        
    
}