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
    public float[] color = new float[4]; // [R,G,B,A]
    public float brightness = 1.0f;
    public float[] position = new float[3]; // [x,y,z]
    public float[] pattern = new float[6]; // [freq, regularity, intensity, symmetry, complexity, variation]

    // constructor - default values 
    public VisualParameters()
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
public class VisualFrame
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
public class VisualizationDataWrapper
{
    public List<UnityVisualData> data;
}

public class SynesthesiaVisualizer : MonoBehaviour
{
    [Header("Data Configuration")]
    public string dataFilePath = "placeholder_visual.json"; //unity_visual_data.json?
    public bool loadOnStart = true;
    public bool enableRealTimeMode = false;

    [Header("Visualization Settings")]
    public GameObject strandPrefab;// w/ audio strand architecture
    public Transform spaceContainer;
    public float updateRate = 60f;
    public float timeScale = 1.0f;
    public bool enableTimeSync = true;

    [Header("Audio Stem Colors")]
    public Color bassColor = Color.blue;
    public Color drumColor = Color.green;

    public Color vocalColor = Color.yellow;
    public Color otherColor = Color.red;


    [Header("Space Domain Settings")]
    public Vector3 spaceBounds = new Vector3(10f, 10f, 10f);
    public float minStrandThickness = 0.1f;
    public float maxStrandThickness = 0.5f;

    [Header("Time Domain Settings")]
    public bool enableFrameCapture = false; //maybe true?
    public float frameDuration = 1.0f;

    [Header("Component Config - Arch101")]
    public Material defaultStrandMaterial;
    public int meshQuality = 16;

    //[Header("")]

    //internal data 

    private List<UnityVisualData> visualDataList;
    private Dictionary<string, AudioStrand> activeStrands;
    private Dictionary<string, List<VisualFrame>> stemFrameBuffers;
    private float elapsedTime = 0f;
    private int currentDataIndex = 0;
    private bool isPlaying = false;

    // frame capture for time domain
    private List<GameObject> currentFrameObjects;
    private float lastFrameTime = 0f;

    //performance monitoring
    private int frameCount = 0;
    private float lastUpdateTime = 0f;

    void Start()
    {
        if (loadOnStart)
        {
            LoadVisualData();
        }
    }

    void Update()
    {
        if (isPlaying && visualDataList != null && visualDataList.Count < 0)
        {
            UpdateVisualization();

        }
        HandleInput();
        UpdatePerformanceMetrics();


    }

    void InitializaVisualization()
    {
        activeStrands = new Dictionary<string, AudioStrand>();
        stemFrameBuffers = new Dictionary<string, List<VisualFrame>>();
        currentFrameObjects = new List<GameObject>();

        // init space container
        if (spaceContainer == null)
        {
            GameObject container = new GameObject("Space Domain");
            spaceContainer = container.transform;
            spaceContainer.position = Vector3.zero;

        }
        // validate prefab architecture
        if (strandPrefab == null)
        {

            CreateStrandPrefab_WithMyArch(); // do this

        }
        else
        {
            ValidateExistingPrefab();

        }
        Debug.Log("Initialized with AudioStrand Architecture");


    }

    void CreateStrandPrefab_WithMyArch()
    {
        Debug.Log("Creating strand prefab modularly");

        strandPrefab = new GameObject("AudioStrandPrefab");

        // add StrandMeshGenerator component
        StrandMeshGenerator meshGen = strandPrefab.AddComponent<StrandMeshGenerator>();
        meshGen.sphereSegments = meshQuality;
        meshGen.sphereRings = meshQuality / 2;
        meshGen.baseRadius = 0.5f;

        // add strandRenderer component
        StrandRenderer renderer = strandPrefab.AddComponent<StrandRenderer>();

        // physics component
        StrandPhysics physics = strandPrefab.AddComponent<StrandPhysics>();
        physics.velocityDamping = 2f;
        physics.position.Lerp = 2f;
        physics.rotationMultiplier = 50f;

        // main AudioStrand coordinator
        AudioStrand strand = strandPrefab.AddComponent<AudioStrand>();
        strand.maxStrandLength = 50f;
        strand.maxStrandPoints = 100;

        //setup default material
        if (defaultStrandMaterial != null)
        {
            MeshRenderer meshRenderer = strandPrefab.AddComponent<MeshRenderer>();
            meshRenderer.material = defaultStrandMaterial;
        }
        strandPrefab.SetActive(false);
        Debug.Log("Created prefab with modular AudioStrand architecture");


    }

    void ValidateExistingPrefab()
    {
        AudioStrand strand = strandPrefab.AddComponent<AudioStrand>();
        if (strand == null)
        {
            Debug.LogError("Strand prefab missing AudioStrand component. Assign prefab.");
            return;
        }

        // check for components
        StrandMeshGenerator meshGen = strandPrefab.GetComponent<StrandMeshGenerator>();
        StrandRenderer renderer = strandPrefab.GetComponent<StrandRenderer>();
        StrandPhysics physics = strandPrefab.GetComponent<StrandPhysics>();

        if (meshGen == null || renderer == null || physics == null)
        {
            if (meshGen == null)
            {
                meshGen = strandPrefab.GetComponent<StrandMeshGenerator>();
                meshGen.sphereSegments = meshQuality;
                meshGen.sphereRings = meshQuality / 2;

            }
            if (renderer == null)
            {
                renderer = strandPrefab.GetComponent<StrandRenderer>();

            }
            if (physics == null)
            {
                physics = strandPrefab.GetComponent<StrandPhysics>();

            }

        }
        Debug.Log("Validated prefab architecture - Mesh, Rend., Physics.");


    }

    public void LoadVisualData()
    {
        string filePath = Path.Combine(Application.streamingAssetsPath, dataFilePath);

        if (!File.Exists(filePath))
        {
            Debug.LogError($"Visual data file not found:{filePath}");
            //search alternate locations

            filePath = Path.Combine(Application.dataPath, "StreamingAssets", dataFilePath);
            if (!File.Exists(filePath))
            {
                Debug.LogError($"Visual data file not found in alternative location: {filePath}");
                CreateFallbackVisualization();
                return;
            }
        }

        try
        {
            string jsonData = File.ReadAllText(filePath);
            // handle other JSON formats from python data generator

            if (jsonData.TrimStart().StartsWith("["))
            {
                //direct array format
                VisualizationDataWrapper wrapper = JsonUtility.FromJson<VisualizationDataWrapper>("{\"data\":" + jsonData + "}");
                visualDataList = wrapper.data;

            }
            else
            {
                //single object format 
                UnityVisualData singleData = JsonUtility.FromJson<UnityVisualData>(jsonData);
                visualDataList = new List<UnityVisualData>(singleData);


            }
            Debug.Log($"Loaded {visualDataList.Count} segments from Unity Data Generator");
            ValidateLoadedData();
            PrepareVisualization();

        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error loading visual data:{e.Message}");
            CreateFallbackVisualization();
        }
    }

    void ValidateLoadedData()
    {
        int totalFrames = 0;

        foreach (var segment in visualDataList)
        {
            if (segment.frames != null)
            {
                totalFrames += segment.frames.Count;

                //validate visual parameters match given python generator's output
                foreach (var frame in segment.frames)
                {
                    if (frame.visual_parameters == null)
                    {
                        frame.visual_parameters = new VisualParameters();
                        Debug.LogWarning($"Missing visual parameters for frame at time {frame.time}, using default instead");
                        
                        
                    }
                }
            }

        }
        Debug.Log($"Validated data: {totalFrames} total frames from NN output.");

        
    }

    void CreateFallbackVisualization()
{
    Debug.Log("Creating fallback visualization data for testing your architecture");
    visualDataList = new List<UnityVisualData>();

    // create test data for test of spectral -> space domain
    UnityVisualData fallbackData = new UnityVisualData
    {
        segment_name = 'architecture_test',
        timestamp = System.DateTime.Now.ToString(),
        frames = new List<VisualFrame>(),

    };
    string[] stems = { "bass", "drums", "vocals", "other" };
    for (int i = 0; i < 100; i++)
    {
        foreach (string stem in stems)
        {
            VisualFrame frame = new VisualFrame
            {
                time = i * 0.1f,
                stem = stem,
                visual_parameters = new VisualParameters()

            };
            // simulate output patterns from NN
            float t = i * 0.1f;
            frame.visual_parameters.shape[0] = 0.3f + 0.4f * Mathf.Sin(t * 0.5f); //roundness
            frame.visual_parameters.shape[1] = 0.2f + 0.3f * Mathf.Sin(t * 0.8f); // complexity
            frame.visual_parameters.position[0] = Mathf.Sin(t) * 2f;
            frame.visual_parameters.position[1] = Mathf.Cos(t) * 2f;
            frame.visual_parameters.brightness = 0.5f + 0.5f * Mathf.Sin(t * 0.3f);

            fallbackData.frames.Add(frame);

        }
    }
    visualDataList.Add(fallbackData);
    PrepareVisualization();
        
    }
    void PrepareVisualization()
{
        // clear existing strands
        foreach( var strand in activeStrands.Values)
        {
            if (strand != null && strand.gameObject != null)
            {
                DestroyImmediate(strand.gameObject);
            }
        }
        activeStrands.Clear();
        stemFrameBuffers.Clear();

        // init srrand buffers for stem types
        string[] stemTypes = {"bass", "drums", "vocal", "other"};

        foreach (string stemType in stemTypes)
        {
            stemFrameBuffers[stemType] = new List<VisualFrame>();

            // instantiate audio strand game obj
            GameObject strandObj = Instantiate(strandPrefab, spaceContainer);
            strandObj.name = $"AudioStrand_{stemType}";
            strandObj.SetActive(true);

            // get AudioStrand component 
            AudioStrand strand = strandObj.GetComponent<AudioStrand>();
            if (strand == null)
            {
                Debug.LogError($"AudioStrand component not found! Check prefab setup.");
                continue;
            }

            // initialize using initialize method
            Color stemColor = GetStemColor(stemType);
            strand.Initialize(stemType, stemColor);
            activeStrands[stemType] = strand;

            Debug.Log($"Created {stemType} strand w/ app architecture");

        }
        // organize frames by stem
        foreach (var segment in visualDataList)
        {
            if (segment.frames != null)
            {
                foreach (var frame in segment.frames)
                {
                    if (stemFrameBuffers.ContainsKey(frame.stem))
                    {
                        stemFrameBuffers[frame.stem].Add(frame);
                    }
                }
            }
        }

        // frame sorting by time
        foreach (var buffer in stemFrameBuffers.Values)
        {
            buffer.Sort((a , b)) => a.Time.CompareTo(b.time));

        }
        Debug.Log("visualization prepared w audio strand architecture");

        foreach( var kvp in stemFrameBuffers)
        {
            Debug.Log($"{kvp.Key}: {kvp.Value.Count} frames")''
        }
    }

    void UpdateVisualization()
    {
        elapsedTime += Time.deltaTime * timeScale;
        frameCount++;

        // update each audio strand using UpdateVisualization()
        foreach ( var stemType in stemFrameBuffers.Keys)
        {
            if (!activeStrands.ContainsKey(stemType)) continue;

            var frames = stemFrameBuffers[stemType];
            var strand = activeStrands[stemType];

            if (strand == null) continue;
            
            // find current frame
            VisualFrame currentFrame = GetFrameAtTime(frames, elapsedTime);
            if (currentFrame != null)
            {
                try 
                {
                    // call updateVisualization() method
                    strand.UpdateVisualization(currentFrame.visual_parameters, elapsedTime);

                }
                catch (System.Exception e)
                {
                    Debug.LogError($"Error updating {stemType} strand: {e.Message}");
                }
            }
        }

        // handle frame capture 
        if (frames.Count == 0) return null;

        // frame lookup for larger data set
        VisualFrame closestFrame = frames[0];
        float closestDistance = Mathf.Abs(closestFrame.time - targetTime);

        foreach (var frame in frames)
        {
            float distance = Mathf.Abs(frame.time - targetTime);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestFrame = frame;

            }

        }
        return closestFrame;


    }

    VisualFrame GetFrameAtTime(List<VisualFrame> frames, float targetTime)
    {
        if (frames.Count == 0) return null;

        // frame lookup for larger data set
        VisualFrame closestFrame = frames[0];
        float closestDistance = Mathf.Abs(closestFrame.time - targetTime);

        foreach (var frame in frames)
        {
            float distance = Mathf.Abs(frame.time - targetTime);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestFrame = frame;
            }
        }
        return closestFrame;
    }

    Color GetStemColor(string stemType)
    {
        switch (stemType.ToLower())
        {
            case "bass": return bassColor;
            case "drums": return drumColor;
            case "vocals": return vocalColor;
            case "other": return otherColor;
            default: return Color.white;
        }
    }

    void CaptureCurrentFrame()
    {
        // live photo implementation
        GameObject frameObj = new GameObject($"Frame_{Time.time:F2}");
        frameObj.transform.SetParent(spaceContainer);

        // capture state of all active strands 
        foreach (var kvp in activeStrands)
        {
            if (kvp.Value == null) continue;

            GameObject strandSnapshot = new GameObject($"Strand_{kvp.Key}");
            strandSnapshot.transform.SetParent(frameObj.transform);
            strandSnapshot.transform.position = kvp.Value.transform.position;
            strandSnapshot.transform.rotation = kvp.Value.transform.rotation;
            strandSnapshot.transform.localScale = kvp.Value.transform.localScale;

            // copy strand component properties
            CopyStrandProperties(kvp.Value.gameObject, strandSnapshot);
        }

        currentFrameObjects.Add(frameObj);

        // limit stored frames for memory management
        if (currentFrameObjects.Count > 10)
        {
            GameObject oldFrame = currentFrameObjects[0];
            currentFrameObjects.RemoveAt(0);
            if (oldFrame != null) Destroy(oldFrame);
        }
    }

    void CopyStrandProperties(GameObject source, GameObject target)
    {
        // copy properties from modular components
        StrandRenderer sourceRenderer = source.GetComponent<StrandRenderer>();
        if (sourceRenderer?.meshRenderer != null)
        {
            MeshRenderer targetRenderer = target.AddComponent<MeshRenderer>();
            targetRenderer.material = new Material(sourceRenderer.meshRenderer.material);

            StrandMeshGenerator sourceMesh = source.GetComponent<StrandMeshGenerator>();
            if (sourceMesh != null)
            {
                MeshFilter targetMesh = target.AddComponent<MeshFilter>();
                MeshFilter sourceMeshFilter = source.GetComponent<MeshFilter>();

                if (sourceMeshFilter?.mesh != null)
                {
                    targetMesh.mesh = sourceMeshFilter.mesh;
                }
            }
        }
    }

    void UpdatePerformanceMetrics()
    {
        if (Time.time - lastUpdateTime >= 1.0f)
        {
            float fps = frameCount / (Time.time - lastUpdateTime);
            if (fps < 30f)
            {
                Debug.LogWarning($"Performance warning: {fps:F1}FPS - consider reducing mesh quality");
            }
            frameCount = 0;
            lastUpdateTime = Time.time;

        }
    }

    void HandleInput()
    {
        if (Input.GetKeyDown(KeyCode.Space))
        {
            TogglePlayback();

        }
        if (Input.GetKeyDown(KeyCode.R))
        {
            ResetVisualization();

        }
        if (Input.GetKeyDown(KeyCode.F))
        {
            enableFrameCapture = !enableFrameCapture;
            Debug.Log($"Time Domain frame capture: {enableFrameCapture}");

        }
        if (Input.GetKeyDown(KeyCode.L))
        {
            LoadVisualData();
        }
        // time scale controls
        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            timeScale = Mathf.Max(0.1f, timeScale - Time.deltaTime * 2f);

        }
        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            timeScale = Mathf.Min(3.0f, timeScale + Time.deltaTime * 2f);

        }
        if (Input.GetKeyDown(KeyCode.T))
        {
            timeScale = 1.0f;
        }

    }

    public void TogglePlayback()
    {
        isPlaying = !isPlaying;
        Debug.Log($"Synth Vis: {(isPlaying ? "Playing" : "Paused" )}");
    }

    public void ResetVisualization()
    {
        elapsedTime = 0f;
        currentDataIndex = 0;

        // reset all strands with ResetStrand() method
        foreach (var strand in activeStrands.Values)
        {
            if (strand != null)
            {
                strand.ResetStrand();

            }
        }

        // Clear Time Domain frame objects
        foreach (var frameObj in currentFrameObjects)
        {
            if (frameObj != null) Destroy(frameObj);
        }

        currentFrameObjects.Clear();
        Debug.Log("Synth Vis Reset");


    }
    
    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 10, 400, 280));

        GUILayout.Label($"Synth Vis - {(isPlaying ? "Playing" : "Paused")}");
        GUILayout.Label($"Modular AudioStrand Architecture");
        GUILayout.Label($"Time: {elapsedTime:F2}s (Scale: {timeScale:F2}).");
        GUILayout.Label($"Active Strands: {activeStrands.Count}");
        GUILayout.Label($"Time Domain Capture: {enableFrameCapture}");


        if (visualDataList != null)
        {
            GUILayout.Label($"Loaded Segments: {visualDataList.Count}");
            int totalFrames = visualDataList.Sum(s => s.frames?.Count ?? 0);
            GUILayout.Label($"Total NN Frames: {totalFrames}");
        }

        GUILayout.Space(10);
        GUILayout.Label("Three Domain Architecture:");
        GUILayout.Label(" ~ SPECTRAL DOMAIN: Neural Network Input/Output");
        GUILayout.Label(" ~ SPACE DOMAIN: Modular Strand Visualization");
        GUILayout.Label(" ~ TIME DOMAIN: Frame Capture System");

        GUILayout.Space(10);
        GUILayout.Label("Controls:");
        GUILayout.Label("Space - Play/Pause");
        GUILayout.Label("R - Reset");
        GUILayout.Label("F - Toggle Time Domain Capture");
        GUILayout.Label("L - Reload Data");
        GUILayout.Label("T - Reset Time Scale");
        GUILayout.Label("← → - Adjust Time Scale");

        GUILayout.EndArea();

    }

    void OnDestroy()
    {
        // Cleanup Time Domain frame objects
        foreach (var frameObj in currentFrameObjects)
        {
            if (frameObj != null) Destroy(frameObj);

        }
        currentFrameObjects.Clear();
    }

    

    


    
}