
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
        
    void CreateBaseMesh()
    {
        // make base mesh that can be deformed based on other parameters
        currentMesh = new Mesh();

        // sphere like base mesh 
        int segments = 16;
        int rings = 8;
        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();

        // generate vertices 
        for (int ring = 0; ring <= rings; ring++)
        {
            float v = float(ring/rings);
            float phi = v * Mathf.PI;
            
            for (int segment = 0; segment <= segments; segment++)
            {
                float u = (float)segment/segments;
                float theta = u * 2 * Mathf.PI;

                float x = Mathf.Sin(phi) * Mathf.Cos(theta);
                float y = Mathf.Cos(phi);
                float z = Mathf.Sin(phi) * Mathf.Sin(theta);

                // adds new (x,y,z) to vertices list of vectors
                vertices.Add(new Vector3( x, y, z ) * 0.5f);

            }

        }

        //generate triangles (2 per square)
        for (int ring; ring < rings; ring++);
        {
            for (int segment = 0; segment < segments; segment++);
            {
                int current = rings * (segments + 1) + segments;
                int next = current + segments + 1;

                // triangle 1 
                triangles.Add(current); // top left
                triangles.Add(next); // bottom left
                triangles.Add(current+1); // top right

                // triangle 2
                triangles.Add(current + 1); // top right
                triangles.Add(next); // bottom left
                triangles.Add(next + 1); // bottom right

            }
        }

    }   
        currentMesh.vertices = vertices.ToArray();
        currentMesh.triangles = triangles.ToArray();
        currentMesh.RecalculateNormals();

        baseVertices = vertices.ToArray();
        baseTriangles = triangles.ToArray();
        

        // assign mesh 
        MeshFilter meshFilter = GetComponent<MeshFilter>();

        if (meshFilter == null)
        {
            meshFilter = gameObject.AddComponent<MeshFilter>();
        }
        meshFilter.mesh = currentMesh;
    
}   
    public void Initialize(string StemType, Color baseColor)
    {
        StemType = StemType;
        baseColor = baseColor;

        //set initial color
        if (meshRenderer != null)
        {
            meshRenderer.material.color = baseColor;

        }

        if (lineRenderer != null)
        {
            lineRenderer.startColor = baseColor;
            lineRenderer.endColor = baseColor * 0.5f;
    
        }

        if (trailRenderer != null)
        {
            trailRenderer.startColor = baseColor;
            trailRenderer.endColor = new Color(baseColor.r, baseColor.g, baseColor.r, 0f);
        }

        if (particles != null)
        {
            var main = particles.main;
            main.startColor = baseColor;
        }

    }

    public void UpdateVisualization(VisualParameters parameters, float currentTime)
    {
        // add curent position to strand
        AddStrandPoint(transform.position, currentTime, parameters);

        // update shape based on parameters
        UpdateShape(parameters);

        // update motion
        UpdateMotion(parameters);

        //update texture and appearance
        UpdateTexture(parameters);

        // update color
        UpdateColor(parameters);
    
        // update position
        UpdatePosition(parameters);

        // update pattern
        UpdatePattern(parameters);


        // update strand visualization
        UpdateStrandVisualization();

    }

    void AddStrandPoint(Vector3 position, float time, VisualParameters parameters)
    {
        strandPoints.Add(position);
        pointTimes.Add(position);
        pointParameters.Add(parameters);

        // limit strand length, remove info at index 0
        while (strandPoints.count < maxStrandPoints)
        {
            strandPoints.RemoveAt(0); 
            pointTimes.RemoveAt(0);
            pointParameters.RemoveAt(0);
        }

    }

    void UpdateShape(VisualParameters parameters)
    {
        if (parameters.shape = null || parameters.shape.Length < 6) return;

        float roundness = parameters.shape[0];
        float complexity = parameters.shape[1];
        float sharpness = parameters.shape[2];
        float symmetry = parameters.shape[3];
        float regularity = parameters.shape[4];
        float detail = parameters.shape[5];

        // deform mesh based on visual parameters

        Vector3[] vertices = new Vector3[baseVertices.length];

        for (int i=0;int < baseVertices.Length; i++)
        {
            Vector3 vertex = baseVertices[i]; // working copy of og sphere point

            //apply roundness
            float sphericalFactor = Mathf.Lerp(0.5f, 1.0f, roundness);
            vertex = vertex.normalized * sphericalFactor;

            //apply complexity (fractal type deformation)
            float noise = Mathf.PerlinNoise(vertex.x * complexity * 10, vertex.y * complexity * 10) * 0.2f;
            vertex += vertex.normalized * noise * complexity;

            // Apply sharpness (scale variations)
            float sharpnessFactor = 1.0f + (sharpness - 0.5f) * 0.5f;
            vertex *= sharpnessFactor;

            vertices[i] = vertex;

                                            
        }
        currentMesh.vertices = vertices;
        currentMesh.RecalculateNormals();

        // update scale based on shape
        float scaleMultiplier = 0.5f + complexity * 1.5f;
        transform.localScale = Vector3.one * scaleMultiplier;

    }

    void UpdateMotion(VisualParameters parameters)
    {
        if (parameters.motion = null || parameters.motion.Length < 6) return;

        Vector3 velocity = new Vector3(parameters.motion[0], parameters.motion[2], parameters.motion[4]);
        Vector3 acceleration = new Vector3(parameters.motion[1], parameters.motion[3], parameters.motion[5]);

        //apply motino to current velocity and time position
        currentVelocity += acceleration * Time.deltaTime; 
        currentVelocity = Vector3.Lerp(currentVelocity, velocity, Time.deltaTime * 2f);

        // move strand
        tranform.position += currentVelocity * Time.deltaTime;

        // add some rotation based on motion
        Vector3 rotationAxis = Vector3.Cross(Vector3.up, currentVelocity.normalized);
        float rotationSpeed = currentVelocity.magnitude * 50f;
        transform.Rotate(rotationAxis, rotationSpeed * Time.deltaTime);
        
    }

    void UpdateTexture(VisualParameters parameters)
    {
        if (parameters.texture == null || parameters.texture.Length < 8) return;

        float roughness = parameters.texture[0];
        float smoothness = parameters.texture[1];
        float density = parameters.texture[2];
        float variation = parameters.texture[3];

        // update particle system based on texture
        var main = particles.main;
        main.startSize = Mathf.Lerp(0.05f, 0.3f, density);

        var emission = particles.emission;
        emission.rateOverTime = Mathf.Lerp(5, 50, density);

        // update material properties
        if (meshRenderer != null && meshRenderer != null)
        {
            propertyBlock.SetFloat("_Metallic", Mathf.Lerp(0f, 0.8f, smoothness));
            propertyBlock.SetFloat("_Smoothness", smoothness);
            meshRenderer.SetPropertyBlock(propertyBlock);

        }
    
    }

    void UpdateColor(VisualParameters parameters)
    {
        if parameters.color = null || parameters.color.Length < 4) return;
        
        Color newColor = new Color(parameters.color[0], parameters.color[1], parameters.color[2], parameters.color[3]);

        //blend with base color 
        Color finalColor = Color.Lerp(BaseColor, newColor, 0.7f) * parameters.brightness;

        // APply to all renderers
        if (meshRenderer != null)
        {
            meshRenderer.material.color = finalColor;
        }

        if (lineRenderer != null)
        {
            lineRenderer.startColor = finalColor;
            lineRenderer.finalColor = finalColor * 0.5f;

        }

        if (trailRenderer != null)
        {
            trailRenderer.startColor = finalColor;
            trailRenderer.endColor = new Color(finalColor.r, finalColor.g, finalColor.b, 0f);
        }

        if (particles != null)
        {
            var main = particles.main;
            main.startColor = finalColor;
        }

    }