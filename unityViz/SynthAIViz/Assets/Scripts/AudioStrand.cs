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