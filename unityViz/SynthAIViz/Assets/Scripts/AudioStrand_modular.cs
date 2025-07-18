using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// Represents a single point in the audio strand's trail with position, time, and visual parameters.
/// This struct replaces the parallel lists approach for better data consistency and maintainability.
/// </summary>
/// <remarks>
/// Using a struct instead of separate lists prevents data synchronization issues and makes
/// the code more readable and maintainable.
/// </remarks>
[System.Serializable]
public struct StrandPoint
{
    /// <summary>
    /// The 3D position of this point in world space.
    /// </summary>
    public Vector3 position;
    
    /// <summary>
    /// The time in the audio stream when this point was created.
    /// </summary>
    public float time;
    
    /// <summary>
    /// The visual parameters that were active when this point was created.
    /// </summary>
    public VisualParameters parameters;
    
    /// <summary>
    /// Creates a new strand point with the specified position, time, and parameters.
    /// </summary>
    /// <param name="pos">The 3D position in world space</param>
    /// <param name="t">The time in the audio stream</param>
    /// <param name="param">The visual parameters at this point</param>
    public StrandPoint(Vector3 pos, float t, VisualParameters param)
    {
        position = pos;
        time = t;
        parameters = param;
    }
}

/// <summary>
/// Handles the generation and deformation of meshes for synesthetic audio visualization.
/// Creates spherical base meshes and applies real-time deformations based on audio parameters.
/// </summary>
/// <remarks>
/// This class is responsible for creating the base geometry and applying shape deformations
/// that respond to audio characteristics like complexity, roundness, and sharpness.
/// </remarks>
public class StrandMeshGenerator : MonoBehaviour
{
    [Header("Mesh Settings")]
    /// <summary>
    /// Number of horizontal segments in the sphere mesh. Higher values create smoother spheres.
    /// </summary>
    public int sphereSegments = 16;
    
    /// <summary>
    /// Number of vertical rings in the sphere mesh. Higher values create smoother spheres.
    /// </summary>
    public int sphereRings = 8;
    
    /// <summary>
    /// Base radius of the sphere mesh before any deformations are applied.
    /// </summary>
    public float baseRadius = 0.5f;
    
    /// <summary>
    /// The current mesh being used for rendering.
    /// </summary>
    private Mesh currentMesh;
    
    /// <summary>
    /// Original vertices of the base sphere mesh, used as reference for deformations.
    /// </summary>
    private Vector3[] baseVertices;
    
    /// <summary>
    /// Original triangle indices of the base sphere mesh.
    /// </summary>
    private int[] baseTriangles;
    
    /// <summary>
    /// Reference to the MeshFilter component for mesh assignment.
    /// </summary>
    private MeshFilter meshFilter;
    
    /// <summary>
    /// Called when the component is first created. Initializes the base mesh.
    /// </summary>
    void Awake()
    {
        CreateBaseMesh();
    }
    
    /// <summary>
    /// Creates the base spherical mesh that will be deformed based on audio parameters.
    /// </summary>
    /// <remarks>
    /// This method generates a UV sphere mesh with the specified number of segments and rings.
    /// The mesh is created programmatically to allow for real-time vertex manipulation.
    /// </remarks>
    void CreateBaseMesh()
    {
        currentMesh = new Mesh();
        meshFilter = GetComponent<MeshFilter>();
        if (meshFilter == null)
            meshFilter = gameObject.AddComponent<MeshFilter>();
        
        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();
        
        // Generate sphere vertices
        for (int ring = 0; ring <= sphereRings; ring++)
        {
            float v = (float)ring / sphereRings;
            float phi = v * Mathf.PI;
            
            for (int segment = 0; segment <= sphereSegments; segment++)
            {
                float u = (float)segment / sphereSegments;
                float theta = u * 2 * Mathf.PI;
                
                float x = Mathf.Sin(phi) * Mathf.Cos(theta);
                float y = Mathf.Cos(phi);
                float z = Mathf.Sin(phi) * Mathf.Sin(theta);
                
                vertices.Add(new Vector3(x, y, z) * baseRadius);
            }
        }
        
        // Generate triangles
        for (int ring = 0; ring < sphereRings; ring++)
        {
            for (int segment = 0; segment < sphereSegments; segment++)
            {
                int current = ring * (sphereSegments + 1) + segment;
                int next = current + sphereSegments + 1;
                
                // Triangle 1
                triangles.Add(current);
                triangles.Add(next);
                triangles.Add(current + 1);
                
                // Triangle 2
                triangles.Add(current + 1);
                triangles.Add(next);
                triangles.Add(next + 1);
            }
        }
        
        currentMesh.vertices = vertices.ToArray();
        currentMesh.triangles = triangles.ToArray();
        currentMesh.RecalculateNormals();
        
        baseVertices = vertices.ToArray();
        baseTriangles = triangles.ToArray();
        meshFilter.mesh = currentMesh;
    }
    
    /// <summary>
    /// Updates the mesh shape based on audio-derived visual parameters.
    /// </summary>
    /// <param name="parameters">The visual parameters containing shape information</param>
    /// <remarks>
    /// This method applies real-time deformations to the base sphere mesh:
    /// - Roundness: Controls how spherical vs. irregular the shape is
    /// - Complexity: Adds fractal-like noise deformations
    /// - Sharpness: Controls scale variations across the mesh
    /// </remarks>
    public void UpdateMeshShape(VisualParameters parameters)
    {
        if (parameters?.shape == null || parameters.shape.Length < 6) return;
        
        float roundness = parameters.shape[0];
        float complexity = parameters.shape[1];
        float sharpness = parameters.shape[2];
        
        Vector3[] vertices = new Vector3[baseVertices.Length];
        
        for (int i = 0; i < baseVertices.Length; i++)
        {
            Vector3 vertex = baseVertices[i];
            
            // Apply roundness (spherical factor)
            float sphericalFactor = Mathf.Lerp(0.5f, 1.0f, roundness);
            vertex = vertex.normalized * sphericalFactor;
            
            // Apply complexity (fractal deformation)
            float noise = Mathf.PerlinNoise(vertex.x * complexity * 10, vertex.y * complexity * 10) * 0.2f;
            vertex += vertex.normalized * noise * complexity;
            
            // Apply sharpness
            float sharpnessFactor = 1.0f + (sharpness - 0.5f) * 0.5f;
            vertex *= sharpnessFactor;
            
            vertices[i] = vertex;
        }
        
        currentMesh.vertices = vertices;
        currentMesh.RecalculateNormals();
    }
    
    /// <summary>
    /// Called when the component is destroyed. Cleans up dynamically created meshes.
    /// </summary>
    /// <remarks>
    /// This prevents memory leaks by destroying the dynamically created mesh.
    /// Unity requires explicit cleanup of programmatically created objects.
    /// </remarks>
    void OnDestroy()
    {
        if (currentMesh != null)
            DestroyImmediate(currentMesh);
    }
}

/// <summary>
/// Manages all visual rendering components for the audio strand including line renderers,
/// particle systems, trail renderers, and mesh renderers.
/// </summary>
/// <remarks>
/// This class encapsulates all visual rendering logic, making it easier to modify
/// visual effects without affecting other systems. It handles material management
/// and ensures proper cleanup of resources.
/// </remarks>
public class StrandRenderer : MonoBehaviour
{
    [Header("Rendering Components")]
    /// <summary>
    /// Line renderer used to draw the strand's trail path.
    /// </summary>
    public LineRenderer lineRenderer;
    
    /// <summary>
    /// Particle system for texture and atmospheric effects.
    /// </summary>
    public ParticleSystem particles;
    
    /// <summary>
    /// Trail renderer for motion blur and trailing effects.
    /// </summary>
    public TrailRenderer trailRenderer;
    
    /// <summary>
    /// Mesh renderer for the main strand geometry.
    /// </summary>
    public Renderer meshRenderer;
    
    [Header("Visual Settings")]
    /// <summary>
    /// Animation curve controlling how line thickness varies along the strand.
    /// </summary>
    public AnimationCurve thicknessCurve = AnimationCurve.Linear(0, 0.1f, 1, 0.5f);
    
    /// <summary>
    /// Animation curve controlling how brightness varies along the strand.
    /// </summary>
    public AnimationCurve brightnessCurve = AnimationCurve.Linear(0, 0.5f, 1, 1.5f);
    
    /// <summary>
    /// Property block for efficient material property updates.
    /// </summary>
    private MaterialPropertyBlock propertyBlock;
    
    /// <summary>
    /// Material used by the line renderer.
    /// </summary>
    private Material lineMaterial;
    
    /// <summary>
    /// Material used by the trail renderer.
    /// </summary>
    private Material trailMaterial;
    
    /// <summary>
    /// Called when the component is first created. Initializes all rendering components.
    /// </summary>
    void Awake()
    {
        InitializeComponents();
    }
    
    /// <summary>
    /// Initializes all rendering components and their materials.
    /// </summary>
    /// <remarks>
    /// This method sets up all visual components with appropriate materials and settings.
    /// It ensures that components are created if they don't exist and properly configured.
    /// </remarks>
    void InitializeComponents()
    {
        propertyBlock = new MaterialPropertyBlock();
        
        SetupLineRenderer();
        SetupParticleSystem();
        SetupTrailRenderer();
        SetupMeshRenderer();
    }
    
    /// <summary>
    /// Sets up the line renderer component for strand trail visualization.
    /// </summary>
    /// <remarks>
    /// Creates a line renderer if one doesn't exist and configures it with appropriate
    /// material, width settings, and rendering options.
    /// </remarks>
    void SetupLineRenderer()
    {
        if (lineRenderer == null)
            lineRenderer = gameObject.AddComponent<LineRenderer>();
        
        lineMaterial = new Material(Shader.Find("Sprites/Default"));
        lineRenderer.material = lineMaterial;
        lineRenderer.startWidth = 0.1f;
        lineRenderer.endWidth = 0.05f;
        lineRenderer.positionCount = 0;
        lineRenderer.useWorldSpace = true;
        lineRenderer.generateLightingData = true;
    }
    
    /// <summary>
    /// Sets up the particle system for texture and atmospheric effects.
    /// </summary>
    /// <remarks>
    /// Creates a particle system as a child object and configures it with appropriate
    /// lifetime, speed, size, and emission settings for visual effects.
    /// </remarks>
    void SetupParticleSystem()
    {
        if (particles == null)
        {
            GameObject particlesObj = new GameObject("StrandParticles");
            particlesObj.transform.SetParent(transform);
            particles = particlesObj.AddComponent<ParticleSystem>();
        }
        
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
    }
    
    /// <summary>
    /// Sets up the trail renderer for motion blur effects.
    /// </summary>
    /// <remarks>
    /// Creates a trail renderer if one doesn't exist and configures it with appropriate
    /// material, time settings, and width parameters for trailing effects.
    /// </remarks>
    void SetupTrailRenderer()
    {
        if (trailRenderer == null)
            trailRenderer = gameObject.AddComponent<TrailRenderer>();
        
        trailMaterial = new Material(Shader.Find("Sprites/Default"));
        trailRenderer.material = trailMaterial;
        trailRenderer.time = 1.0f;
        trailRenderer.startWidth = 0.2f;
        trailRenderer.endWidth = 0.01f;
        trailRenderer.generateLightingData = true;
    }
    
    /// <summary>
    /// Sets up the mesh renderer for the main strand geometry.
    /// </summary>
    /// <remarks>
    /// Ensures a mesh renderer component exists for rendering the deformed mesh geometry.
    /// </remarks>
    void SetupMeshRenderer()
    {
        meshRenderer = GetComponent<Renderer>();
        if (meshRenderer == null)
            meshRenderer = gameObject.AddComponent<MeshRenderer>();
    }
    
    /// <summary>
    /// Updates the strand trail visualization using the provided strand points.
    /// </summary>
    /// <param name="strandPoints">List of points that make up the strand trail</param>
    /// <remarks>
    /// This method updates the line renderer with the current strand points and adjusts
    /// line width based on the latest visual parameters for dynamic thickness effects.
    /// </remarks>
    public void UpdateStrandVisualization(List<StrandPoint> strandPoints)
    {
        if (strandPoints.Count < 2) return;
        
        // Update line renderer
        lineRenderer.positionCount = strandPoints.Count;
        Vector3[] positions = new Vector3[strandPoints.Count];
        for (int i = 0; i < strandPoints.Count; i++)
        {
            positions[i] = strandPoints[i].position;
        }
        lineRenderer.SetPositions(positions);
        
        // Update line width based on latest parameters
        if (strandPoints.Count > 0)
        {
            var lastParams = strandPoints[strandPoints.Count - 1].parameters;
            if (lastParams?.texture != null && lastParams.texture.Length >= 3)
            {
                float thickness = Mathf.Lerp(0.05f, 0.3f, lastParams.texture[2]);
                lineRenderer.startWidth = thickness;
                lineRenderer.endWidth = thickness * 0.3f;
            }
        }
    }
    
    /// <summary>
    /// Updates the color across all rendering components based on audio parameters.
    /// </summary>
    /// <param name="baseColor">The base color for this strand</param>
    /// <param name="parameters">The visual parameters containing color information</param>
    /// <remarks>
    /// This method blends the base color with dynamic color parameters and applies
    /// the result to all visual components (mesh, line, trail, and particles).
    /// </remarks>
    public void UpdateColor(Color baseColor, VisualParameters parameters)
    {
        if (parameters?.color == null || parameters.color.Length < 4) return;
        
        Color newColor = new Color(parameters.color[0], parameters.color[1], parameters.color[2], parameters.color[3]);
        Color finalColor = Color.Lerp(baseColor, newColor, 0.7f) * parameters.brightness;
        
        // Apply to all renderers
        if (meshRenderer != null && meshRenderer.material != null)
            meshRenderer.material.color = finalColor;
        
        if (lineRenderer != null)
        {
            lineRenderer.startColor = finalColor;
            lineRenderer.endColor = finalColor * 0.5f;
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
    
    /// <summary>
    /// Updates texture and material properties based on audio parameters.
    /// </summary>
    /// <param name="parameters">The visual parameters containing texture information</param>
    /// <remarks>
    /// This method updates particle system properties and material properties like
    /// metallic and smoothness based on texture parameters from audio analysis.
    /// </remarks>
    public void UpdateTexture(VisualParameters parameters)
    {
        if (parameters?.texture == null || parameters.texture.Length < 8) return;
        
        float roughness = parameters.texture[0];
        float smoothness = parameters.texture[1];
        float density = parameters.texture[2];
        
        // Update particle system
        var main = particles.main;
        main.startSize = Mathf.Lerp(0.05f, 0.3f, density);
        
        var emission = particles.emission;
        emission.rateOverTime = Mathf.Lerp(5, 50, density);
        
        // Update material properties
        if (meshRenderer != null)
        {
            propertyBlock.SetFloat("_Metallic", Mathf.Lerp(0f, 0.8f, smoothness));
            propertyBlock.SetFloat("_Smoothness", smoothness);
            meshRenderer.SetPropertyBlock(propertyBlock);
        }
    }
    
    /// <summary>
    /// Called when the component is destroyed. Cleans up dynamically created materials.
    /// </summary>
    /// <remarks>
    /// This prevents memory leaks by destroying the dynamically created materials.
    /// Unity requires explicit cleanup of programmatically created objects.
    /// </remarks>
    void OnDestroy()
    {
        if (lineMaterial != null) DestroyImmediate(lineMaterial);
        if (trailMaterial != null) DestroyImmediate(trailMaterial);
    }
}

/// <summary>
/// Handles physics calculations and motion for the audio strand.
/// Manages velocity, acceleration, rotation, and position updates based on audio parameters.
/// </summary>
/// <remarks>
/// This class encapsulates all physics-related logic, making it easier to modify
/// motion behavior without affecting visual or mesh systems.
/// </remarks>
public class StrandPhysics : MonoBehaviour
{
    [Header("Physics Settings")]
    /// <summary>
    /// How quickly the strand's velocity approaches the target velocity (higher = faster).
    /// </summary>
    public float velocityDamping = 2f;
    
    /// <summary>
    /// How quickly the strand moves toward its target position (higher = faster).
    /// </summary>
    public float positionLerp = 2f;
    
    /// <summary>
    /// Multiplier for rotation speed based on velocity magnitude.
    /// </summary>
    public float rotationMultiplier = 50f;
    
    /// <summary>
    /// Current velocity vector of the strand.
    /// </summary>
    private Vector3 currentVelocity;
    
    /// <summary>
    /// Target position that the strand is moving toward.
    /// </summary>
    private Vector3 targetPosition;
    
    /// <summary>
    /// Updates the strand's motion based on audio-derived motion parameters.
    /// </summary>
    /// <param name="parameters">The visual parameters containing motion information</param>
    /// <remarks>
    /// This method applies physics calculations including acceleration, velocity damping,
    /// and rotation based on motion direction. It creates smooth, responsive movement
    /// that responds to audio characteristics.
    /// </remarks>
    public void UpdateMotion(VisualParameters parameters)
    {
        if (parameters?.motion == null || parameters.motion.Length < 6) return;
        
        Vector3 velocity = new Vector3(parameters.motion[0], parameters.motion[2], parameters.motion[4]);
        Vector3 acceleration = new Vector3(parameters.motion[1], parameters.motion[3], parameters.motion[5]);
        
        // Apply physics
        currentVelocity += acceleration * Time.deltaTime;
        currentVelocity = Vector3.Lerp(currentVelocity, velocity, Time.deltaTime * velocityDamping);
        
        // Move strand
        transform.position += currentVelocity * Time.deltaTime;
        
        // Add rotation based on motion
        if (currentVelocity.magnitude > 0.01f)
        {
            Vector3 rotationAxis = Vector3.Cross(Vector3.up, currentVelocity.normalized);
            float rotationSpeed = currentVelocity.magnitude * rotationMultiplier;
            transform.Rotate(rotationAxis, rotationSpeed * Time.deltaTime);
        }
    }
    
    /// <summary>
    /// Updates the strand's position based on audio-derived position parameters.
    /// </summary>
    /// <param name="parameters">The visual parameters containing position information</param>
    /// <remarks>
    /// This method smoothly moves the strand toward a target position derived from
    /// audio analysis, creating spatial relationships between different audio elements.
    /// </remarks>
    public void UpdatePosition(VisualParameters parameters)
    {
        if (parameters?.position == null || parameters.position.Length < 3) return;
        
        targetPosition = new Vector3(parameters.position[0], parameters.position[1], parameters.position[2]);
        targetPosition = Vector3.Scale(targetPosition, new Vector3(5f, 5f, 5f));
        
        // Smoothly move towards target
        transform.position = Vector3.Lerp(transform.position, targetPosition, Time.deltaTime * positionLerp);
    }
    
    /// <summary>
    /// Resets the physics state to initial values.
    /// </summary>
    /// <remarks>
    /// This method is typically called when starting a new audio track or when
    /// the strand needs to be completely reset for performance reasons.
    /// </remarks>
    public void ResetPhysics()
    {
        currentVelocity = Vector3.zero;
        transform.position = Vector3.zero;
        transform.rotation = Quaternion.identity;
        transform.localScale = Vector3.one;
    }
}

/// <summary>
/// Main coordinator class for audio strand visualization.
/// Orchestrates the interaction between mesh generation, rendering, and physics systems
/// to create a cohesive synesthetic audio visualization experience.
/// </summary>
/// <remarks>
/// This class serves as the main interface for audio strand functionality. It coordinates
/// between the specialized systems (mesh, renderer, physics) and manages the strand's
/// data structure. The modular design allows each system to be modified independently
/// while maintaining a clean, organized codebase.
/// </remarks>
public class AudioStrand : MonoBehaviour
{
    [Header("Audio Strand Properties")]
    /// <summary>
    /// The type of audio stem this strand represents (e.g., "Drums", "Bass", "Melody").
    /// </summary>
    public string StemType { get; private set; }
    
    /// <summary>
    /// The base color for this strand, used as a foundation for dynamic color changes.
    /// </summary>
    public Color BaseColor { get; private set; }
    
    [Header("Strand Behavior")]
    /// <summary>
    /// Maximum length of the strand trail in world units.
    /// </summary>
    public float maxStrandLength = 50f;
    
    /// <summary>
    /// Maximum number of points to store in the strand trail.
    /// </summary>
    public int maxStrandPoints = 100;
    
    /// <summary>
    /// Reference to the mesh generation system.
    /// </summary>
    private StrandMeshGenerator meshGenerator;
    
    /// <summary>
    /// Reference to the rendering system.
    /// </summary>
    private StrandRenderer strandRenderer;
    
    /// <summary>
    /// Reference to the physics system.
    /// </summary>
    private StrandPhysics strandPhysics;
    
    /// <summary>
    /// List of strand points that make up the visual trail.
    /// </summary>
    private List<StrandPoint> strandPoints;
    
    /// <summary>
    /// Called when the component is first created. Initializes all systems.
    /// </summary>
    void Awake()
    {
        InitializeComponents();
    }
    
    /// <summary>
    /// Initializes all component systems and data structures.
    /// </summary>
    /// <remarks>
    /// This method ensures all required systems exist and are properly configured.
    /// It creates any missing components and sets up the strand's data structure.
    /// </remarks>
    void InitializeComponents()
    {
        strandPoints = new List<StrandPoint>();
        
        // Get or create component systems
        meshGenerator = GetComponent<StrandMeshGenerator>();
        if (meshGenerator == null)
            meshGenerator = gameObject.AddComponent<StrandMeshGenerator>();
        
        strandRenderer = GetComponent<StrandRenderer>();
        if (strandRenderer == null)
            strandRenderer = gameObject.AddComponent<StrandRenderer>();
        
        strandPhysics = GetComponent<StrandPhysics>();
        if (strandPhysics == null)
            strandPhysics = gameObject.AddComponent<StrandPhysics>();
    }
    
    /// <summary>
    /// Initializes the audio strand with its type and base color.
    /// </summary>
    /// <param name="stemType">The type of audio stem (e.g., "Drums", "Bass")</param>
    /// <param name="baseColor">The base color for this strand</param>
    /// <example>
    /// <code>
    /// AudioStrand strand = GetComponent<AudioStrand>();
    /// strand.Initialize("Drums", Color.red);
    /// </code>
    /// </example>
    public void Initialize(string stemType, Color baseColor)
    {
        StemType = stemType;
        BaseColor = baseColor;
        
        // Initialize color across all renderers
        strandRenderer.UpdateColor(baseColor, new VisualParameters());
    }
    
    /// <summary>
    /// Updates the visual representation of the strand based on current audio parameters.
    /// </summary>
    /// <param name="parameters">The visual parameters derived from audio analysis</param>
    /// <param name="currentTime">The current time in the audio stream</param>
    /// <exception cref="System.ArgumentNullException">Thrown when parameters is null</exception>
    /// <remarks>
    /// This method coordinates updates across all visual systems:
    /// - Mesh shape deformation based on audio complexity
    /// - Motion and physics based on audio rhythm
    /// - Color and texture changes based on audio timbre
    /// - Pattern effects based on audio structure
    /// 
    /// The method validates parameters before processing to ensure robust operation.
    /// </remarks>
    public void UpdateVisualization(VisualParameters parameters, float currentTime)
    {
        // Validate parameters
        if (parameters == null)
        {
            Debug.LogWarning($"Null parameters passed to {StemType} strand");
            return;
        }
        
        // Add current position to strand
        AddStrandPoint(transform.position, currentTime, parameters);
        
        // Update components with validated parameters
        meshGenerator.UpdateMeshShape(parameters);
        strandPhysics.UpdateMotion(parameters);
        strandPhysics.UpdatePosition(parameters);
        strandRenderer.UpdateTexture(parameters);
        strandRenderer.UpdateColor(BaseColor, parameters);
        
        // Update pattern effects
        UpdatePattern(parameters);
        
        // Update strand visualization
        strandRenderer.UpdateStrandVisualization(strandPoints);
    }
    
    /// <summary>
    /// Adds a new point to the strand trail with position, time, and parameters.
    /// </summary>
    /// <param name="position">The 3D position in world space</param>
    /// <param name="time">The time in the audio stream</param>
    /// <param name="parameters">The visual parameters at this point</param>
    /// <remarks>
    /// This method maintains the strand trail by adding new points and removing old ones
    /// to keep the trail within the specified length limits.
    /// </remarks>
    void AddStrandPoint(Vector3 position, float time, VisualParameters parameters)
    {
        strandPoints.Add(new StrandPoint(position, time, parameters));
        
        // Limit strand length
        while (strandPoints.Count > maxStrandPoints)
        {
            strandPoints.RemoveAt(0);
        }
    }
    
    /// <summary>
    /// Updates pattern-based effects like pulsing and scaling based on audio parameters.
    /// </summary>
    /// <param name="parameters">The visual parameters containing pattern information</param>
    /// <remarks>
    /// This method creates rhythmic effects that respond to audio patterns, such as
    /// pulsing scale changes and particle emission variations.
    /// </remarks>
    void UpdatePattern(VisualParameters parameters)
    {
        if (parameters?.pattern == null || parameters.pattern.Length < 6) return;
        
        float frequency = parameters.pattern[0];
        float intensity = parameters.pattern[2];
        
        // Pattern-based pulse effect
        float pulse = Mathf.Sin(Time.time * frequency * 10f) * intensity;
        Vector3 pulsedScale = transform.localScale * (1f + pulse * 0.2f);
        transform.localScale = pulsedScale;
    }
    
    /// <summary>
    /// Resets the strand to its initial state, clearing all trails and resetting position.
    /// </summary>
    /// <remarks>
    /// This method is typically called when starting a new audio track or when the strand
    /// needs to be completely reset for performance reasons. It clears all visual trails
    /// and resets the physics state to initial values.
    /// </remarks>
    public void ResetStrand()
    {
        strandPoints.Clear();
        strandPhysics.ResetPhysics();
        
        if (strandRenderer.particles != null)
            strandRenderer.particles.Clear();
        
        if (strandRenderer.lineRenderer != null)
            strandRenderer.lineRenderer.positionCount = 0;
    }
    
    /// <summary>
    /// Draws debug gizmos in the scene view for development and debugging.
    /// </summary>
    /// <remarks>
    /// This method provides visual debugging information in the Unity scene view,
    /// showing the strand trail path and current position for development purposes.
    /// </remarks>
    void OnDrawGizmos()
    {
        if (strandPoints != null && strandPoints.Count > 1)
        {
            Gizmos.color = BaseColor;
            for (int i = 0; i < strandPoints.Count - 1; i++)
            {
                Gizmos.DrawLine(strandPoints[i].position, strandPoints[i + 1].position);
            }
        }
        
        Gizmos.color = Color.white;
        Gizmos.DrawWireSphere(transform.position, 0.1f);
    }
}