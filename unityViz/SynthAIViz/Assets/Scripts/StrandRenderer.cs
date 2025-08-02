using UnityEngine;
using System.Collections.Generic;

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