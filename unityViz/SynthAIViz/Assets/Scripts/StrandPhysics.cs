using UnityEngine;

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