using UnityEngine;
using System.Collections.Generic;

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