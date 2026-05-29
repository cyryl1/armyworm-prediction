import React, { useState, useEffect } from 'react';
import * as SecureStore from 'expo-secure-store';
import {
  StyleSheet,
  View,
  Text,
  Pressable,
  ActivityIndicator,
  ScrollView,
  TextInput,
  Alert,
  Platform,
  TouchableOpacity,
  Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import * as ImagePicker from 'expo-image-picker';
import { Image } from 'expo-image';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { useTheme } from '@/hooks/use-theme';
import {
  loadApiSettings,
  saveApiKey,
  saveApiUrl,
} from '@/lib/api-settings';

// Custom type matching backend schemas
interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  bbox: [number, number, number, number];
  recommendation: string;
  recommendation_details?: string | null;
  gps_latitude?: number | null;
  gps_longitude?: number | null;
  detection_timestamp: string;
}

interface DetectionResponse {
  detections: Detection[];
  annotated_image: string | null;
}

export default function HomeScreen() {
  const theme = useTheme();
  
  // App state
  const [apiUrl, setApiUrl] = useState<string>(
    Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000'
  );
  const [apiKey, setApiKey] = useState<string>('');
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [serverStatus, setServerStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [modelLoaded, setModelLoaded] = useState<boolean>(false);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [scanPulse, setScanPulse] = useState<boolean>(false);

  // Check server health
  const checkServerHealth = async (customUrl?: string, customApiKey?: string) => {
    const urlToCheck = customUrl || apiUrl;
    const keyToUse = customApiKey ?? apiKey;
    setServerStatus('checking');
    try {
      const response = await fetch(`${urlToCheck}/health`, {
        headers: { 'accept': 'application/json', ...(keyToUse ? { 'x-api-key': keyToUse } : {}) },
      });
      if (response.ok) {
        const data = await response.json();
        setServerStatus('online');
        setModelLoaded(!!data.model_loaded);
      } else {
        setServerStatus('offline');
        setModelLoaded(false);
      }
    } catch (error) {
      setServerStatus('offline');
      setModelLoaded(false);
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const settings = await loadApiSettings();
        setApiUrl(settings.apiUrl);
        setApiKey(settings.apiKey || '');
        await checkServerHealth(settings.apiUrl, settings.apiKey || undefined);
      } catch (e) {
        await checkServerHealth();
      }
    })();
  }, []);

  // Set scanning line animation intervals
  useEffect(() => {
    let interval: any;
    if (isAnalyzing) {
      interval = setInterval(() => {
        setScanPulse(prev => !prev);
      }, 1000);
    } else {
      setScanPulse(false);
    }
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  // Request camera and library permissions
  const requestPermissions = async () => {
    if (Platform.OS !== 'web') {
      const libraryStatus = await ImagePicker.requestMediaLibraryPermissionsAsync();
      const cameraStatus = await ImagePicker.requestCameraPermissionsAsync();
      
      if (libraryStatus.status !== 'granted' || cameraStatus.status !== 'granted') {
        Alert.alert(
          'Permissions Required',
          'Sorry, we need camera and library roll permissions to take and upload leaf images!'
        );
        return false;
      }
      return true;
    }
    return true;
  };

  // Capture Image
  const takePhoto = async () => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    try {
      const result = await ImagePicker.launchCameraAsync({
        mediaTypes: ['images'],
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.9,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        setImageUri(result.assets[0].uri);
        setResult(null);
        analyzeImage(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to capture photo');
    }
  };

  // Upload Image
  const pickImage = async () => {
    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.9,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        setImageUri(result.assets[0].uri);
        setResult(null);
        analyzeImage(result.assets[0].uri);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to select image');
    }
  };

  // Trigger Backend API Request
  const analyzeImage = async (uri: string) => {
    setIsAnalyzing(true);
    
    // Create multipart form data
    const formData = new FormData();
    const filename = uri.split('/').pop() || 'leaf.jpg';
    const match = /\.(\w+)$/.exec(filename);
    const type = match ? `image/${match[1]}` : `image/jpeg`;
    
    formData.append('file', {
      uri: Platform.OS === 'ios' ? uri.replace('file://', '') : uri,
      name: filename,
      type,
    } as any);

    try {
      // Endpoint requires a temporary API key for headers if configured
      // In this setup, we can pass a dummy header or let dependency injection handle it
      const response = await fetch(`${apiUrl}/detect`, {
        method: 'POST',
        body: formData,
        headers: {
            'accept': 'application/json',
            'Content-Type': 'multipart/form-data',
            ...(apiKey ? { 'x-api-key': apiKey } : {}),
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Server returned an error');
      }

      const data: DetectionResponse = await response.json();
      setResult(data);
    } catch (error: any) {
      Alert.alert(
        'Analysis Failed',
        error?.message || 'Failed to connect to the AI server. Please check your API URL in settings.'
      );
      setImageUri(null);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Clear current scans
  const resetScan = () => {
    setImageUri(null);
    setResult(null);
  };

  // Category Color Codes
  const getCategoryColor = (className: string) => {
    const name = className.toLowerCase();
    if (name.includes('healthy')) return '#10b981'; // Emerald Green
    if (name.includes('larva') || name.includes('egg')) return '#ef4444'; // Crimson Red
    if (name.includes('damage') || name.includes('frass')) return '#f97316'; // Orange Warning
    return '#3b82f6'; // Blue Default
  };

  const getCategoryIcon = (className: string) => {
    const name = className.toLowerCase();
    if (name.includes('healthy')) return 'check-circle-outline';
    if (name.includes('larva')) return 'bug-outline';
    if (name.includes('egg')) return 'ellipse-outline';
    if (name.includes('damage')) return 'alert-decagram-outline';
    return 'image-filter-center-focus';
  };

  return (
    <ThemedView style={styles.container}>
      <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
        {/* Header Section */}
        <View style={styles.header}>
          <View>
            <ThemedText type="subtitle" style={styles.headerTitle}>
              Pest Detect AI
            </ThemedText>
            <View style={styles.statusBadge}>
              <View
                style={[
                  styles.statusIndicator,
                  {
                    backgroundColor:
                      serverStatus === 'online'
                        ? modelLoaded
                          ? '#10b981' // Solid Green
                          : '#eab308' // Yellow
                        : '#ef4444', // Red
                  },
                ]}
              />
              <ThemedText type="small" themeColor="textSecondary" style={styles.statusText}>
                {serverStatus === 'checking' && 'Connecting to AI...'}
                {serverStatus === 'online' &&
                  (modelLoaded ? 'AI Server Online' : 'AI Model Loading...')}
                {serverStatus === 'offline' && 'AI Server Offline'}
              </ThemedText>
            </View>
          </View>
          
          <TouchableOpacity
            onPress={() => setShowSettings(!showSettings)}
            style={[styles.settingsButton, { backgroundColor: theme.backgroundElement }]}>
            <Ionicons name="settings-outline" size={22} color={theme.text} />
          </TouchableOpacity>
        </View>

        <ScrollView
          showsVerticalScrollIndicator={false}
          contentContainerStyle={styles.scrollContent}>
          
          {/* Settings Accordion */}
          {showSettings && (
            <ThemedView type="backgroundElement" style={styles.settingsCard}>
              <View style={styles.settingsHeader}>
                <ThemedText type="smallBold">API Server Configuration</ThemedText>
                <TouchableOpacity onPress={() => checkServerHealth()}>
                  <Ionicons name="refresh-outline" size={16} color={theme.textSecondary} />
                </TouchableOpacity>
              </View>
              <TextInput
                style={[
                  styles.apiInput,
                  { color: theme.text, borderColor: theme.backgroundSelected },
                ]}
                value={apiUrl}
                onChangeText={async (v) => {
                  setApiUrl(v);
                  try {
                    await saveApiUrl(v);
                  } catch (e) {}
                }}
                placeholder="http://localhost:8000"
                placeholderTextColor={theme.textSecondary}
                autoCapitalize="none"
                autoCorrect={false}
              />
              <ThemedText type="small" style={{ marginTop: 8 }}>API Key (optional)</ThemedText>
              <TextInput
                style={[
                  styles.apiInput,
                  { color: theme.text, borderColor: theme.backgroundSelected },
                ]}
                value={apiKey}
                onChangeText={async (v) => {
                  setApiKey(v);
                  try {
                    await saveApiKey(v);
                  } catch (e) {
                    // ignore
                  }
                }}
                placeholder="Enter x-api-key header if required"
                placeholderTextColor={theme.textSecondary}
                autoCapitalize="none"
                autoCorrect={false}
              />
              <ThemedText type="code" style={styles.hintText}>
                * iOS: http://localhost:8000 {'\n'}
                * Android: http://10.0.2.2:8000 {'\n'}
                * Mobile Device: http://[your-pc-ip]:8000
              </ThemedText>
            </ThemedView>
          )}

          {/* Core Image Scan Display */}
          {!imageUri ? (
            <TouchableOpacity
              activeOpacity={0.9}
              onPress={pickImage}
              style={[
                styles.uploadCard,
                {
                  backgroundColor: theme.backgroundElement,
                  borderColor: theme.backgroundSelected,
                },
              ]}>
              <View style={styles.uploadCardInner}>
                <View style={[styles.uploadIconContainer, { backgroundColor: theme.backgroundSelected }]}>
                  <MaterialCommunityIcons name="leaf" size={44} color="#10b981" />
                </View>
                <ThemedText type="default" style={styles.uploadTitle}>
                  Scan Maize Leaf
                </ThemedText>
                <ThemedText themeColor="textSecondary" style={styles.uploadSubtitle}>
                  Take a photo or upload an image to identify fall armyworms and maize diseases instantly.
                </ThemedText>
                
                <View style={styles.uploadActions}>
                  <TouchableOpacity style={styles.actionBtnPrimary} onPress={takePhoto}>
                    <Ionicons name="camera" size={18} color="#fff" />
                    <Text style={styles.actionBtnText}>Take Photo</Text>
                  </TouchableOpacity>
                  
                  <TouchableOpacity
                    style={[styles.actionBtnSecondary, { backgroundColor: theme.backgroundSelected }]}
                    onPress={pickImage}>
                    <Ionicons name="image" size={18} color={theme.text} />
                    <Text style={[styles.actionBtnTextSecondary, { color: theme.text }]}>Upload</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </TouchableOpacity>
          ) : (
            <View style={styles.previewContainer}>
              <View style={styles.imageCard}>
                {/* Check if we have annotated image from the API, else show local picked image */}
                <Image
                  source={{
                    uri: result?.annotated_image
                      ? `data:image/jpeg;base64,${result.annotated_image}`
                      : imageUri,
                  }}
                  style={styles.previewImage}
                  contentFit="cover"
                />
                
                {/* Glow Scanner Bar during AI inference */}
                {isAnalyzing && (
                  <View
                    style={[
                      styles.scannerLine,
                      { top: scanPulse ? '90%' : '10%' },
                    ]}
                  />
                )}
                
                {isAnalyzing && (
                  <View style={styles.loaderOverlay}>
                    <ActivityIndicator size="large" color="#10b981" />
                    <Text style={styles.loaderText}>AI Model Processing...</Text>
                  </View>
                )}
              </View>

              {/* Action Buttons below Preview */}
              {!isAnalyzing && (
                <View style={styles.previewActions}>
                  <TouchableOpacity
                    style={[styles.previewActionBtn, { backgroundColor: '#ef4444' }]}
                    onPress={resetScan}>
                    <Ionicons name="trash-outline" size={18} color="#fff" />
                    <Text style={styles.previewActionBtnText}>Clear Scan</Text>
                  </TouchableOpacity>
                  
                  <TouchableOpacity
                    style={[styles.previewActionBtn, { backgroundColor: '#10b981' }]}
                    onPress={pickImage}>
                    <Ionicons name="refresh-outline" size={18} color="#fff" />
                    <Text style={styles.previewActionBtnText}>Scan Another</Text>
                  </TouchableOpacity>
                </View>
              )}

              {/* Detection Results */}
              {!isAnalyzing && result && (
                <View style={styles.resultsWrapper}>
                  <ThemedText type="smallBold" style={styles.sectionTitle}>
                    Diagnostic Detections ({result.detections.length})
                  </ThemedText>

                  {result.detections.length === 0 ? (
                    <ThemedView type="backgroundElement" style={styles.healthyCard}>
                      <Ionicons name="checkmark-circle" size={40} color="#10b981" />
                      <View style={styles.healthyInfo}>
                        <ThemedText type="default" style={styles.healthyTitle}>
                          Crop Healthy!
                        </ThemedText>
                        <ThemedText themeColor="textSecondary" type="small">
                          No fall armyworms or leaf diseases were found in this maize leaf scan.
                        </ThemedText>
                      </View>
                    </ThemedView>
                  ) : (
                    result.detections.map((detection, index) => {
                      const badgeColor = getCategoryColor(detection.class_name);
                      const iconName = getCategoryIcon(detection.class_name);

                      return (
                        <ThemedView
                          key={index}
                          type="backgroundElement"
                          style={[styles.detectionCard, { borderLeftColor: badgeColor }]}>
                          <View style={styles.cardHeader}>
                            <View style={styles.cardTitleContainer}>
                              <MaterialCommunityIcons
                                name={iconName as any}
                                size={22}
                                color={badgeColor}
                              />
                              <ThemedText type="default" style={styles.detectionClassName}>
                                {detection.class_name
                                  .replace(/-/g, ' ')
                                  .replace(/\b\w/g, c => c.toUpperCase())}
                              </ThemedText>
                            </View>
                            
                            <View style={[styles.confidenceBadge, { backgroundColor: badgeColor + '20' }]}>
                              <Text style={[styles.confidenceText, { color: badgeColor }]}>
                                {Math.round(detection.confidence * 100)}% Match
                              </Text>
                            </View>
                          </View>

                          {/* Confidence Bar */}
                          <View style={styles.progressBarBg}>
                            <View
                              style={[
                                styles.progressBarFill,
                                {
                                  backgroundColor: badgeColor,
                                  width: `${detection.confidence * 100}%`,
                                },
                              ]}
                            />
                          </View>

                          {/* Recommendation Section */}
                          <View style={styles.recommendationCard}>
                            <ThemedText type="smallBold" style={styles.recommendationLabel}>
                              AI Management Recommendation:
                            </ThemedText>
                            <ThemedText type="small" themeColor="text">
                              {detection.recommendation}
                            </ThemedText>
                            {detection.recommendation_details && (
                              <ThemedText
                                type="small"
                                themeColor="textSecondary"
                                style={styles.recommendationDetails}>
                                {detection.recommendation_details}
                              </ThemedText>
                            )}
                          </View>

                          {/* GPS Metadata */}
                          {(detection.gps_latitude || detection.gps_longitude) && (
                            <View style={styles.locationMetadata}>
                              <Ionicons name="location-outline" size={12} color={theme.textSecondary} />
                              <ThemedText type="code" themeColor="textSecondary" style={styles.locationText}>
                                Coordinates: {detection.gps_latitude?.toFixed(4)}, {detection.gps_longitude?.toFixed(4)}
                              </ThemedText>
                            </View>
                          )}
                        </ThemedView>
                      );
                    })
                  )}
                </View>
              )}
            </View>
          )}
        </ScrollView>
      </SafeAreaView>
    </ThemedView>
  );
}

const screenWidth = Dimensions.get('window').width;

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: Spacing.four,
    paddingVertical: Spacing.three,
  },
  headerTitle: {
    fontSize: 24,
    lineHeight: 28,
    fontWeight: 'bold',
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: Spacing.half,
  },
  statusIndicator: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: Spacing.one,
  },
  statusText: {
    fontSize: 12,
  },
  settingsButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  scrollContent: {
    paddingHorizontal: Spacing.four,
    paddingBottom: BottomTabInset + Spacing.five,
    maxWidth: MaxContentWidth,
    alignSelf: 'center',
    width: '100%',
  },
  settingsCard: {
    borderRadius: Spacing.three,
    padding: Spacing.three,
    marginBottom: Spacing.four,
  },
  settingsHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.two,
  },
  apiInput: {
    height: 44,
    borderWidth: 1,
    borderRadius: Spacing.two,
    paddingHorizontal: Spacing.three,
    fontSize: 14,
  },
  hintText: {
    fontSize: 11,
    marginTop: Spacing.two,
    lineHeight: 16,
  },
  uploadCard: {
    borderRadius: Spacing.four,
    borderWidth: 2,
    borderStyle: 'dashed',
    padding: Spacing.five,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: Spacing.four,
    minHeight: 340,
  },
  uploadCardInner: {
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
  },
  uploadIconContainer: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: Spacing.four,
  },
  uploadTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: Spacing.two,
    textAlign: 'center',
  },
  uploadSubtitle: {
    textAlign: 'center',
    fontSize: 14,
    lineHeight: 20,
    paddingHorizontal: Spacing.two,
    marginBottom: Spacing.five,
  },
  uploadActions: {
    flexDirection: 'row',
    gap: Spacing.three,
    width: '100%',
    justifyContent: 'center',
  },
  actionBtnPrimary: {
    backgroundColor: '#10b981',
    flexDirection: 'row',
    paddingVertical: Spacing.two + 2,
    paddingHorizontal: Spacing.four,
    borderRadius: Spacing.three,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.two,
    flex: 1,
    maxWidth: 160,
  },
  actionBtnText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  actionBtnSecondary: {
    flexDirection: 'row',
    paddingVertical: Spacing.two + 2,
    paddingHorizontal: Spacing.four,
    borderRadius: Spacing.three,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.two,
    flex: 1,
    maxWidth: 160,
  },
  actionBtnTextSecondary: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  previewContainer: {
    marginTop: Spacing.three,
  },
  imageCard: {
    width: '100%',
    aspectRatio: 4 / 3,
    borderRadius: Spacing.four,
    overflow: 'hidden',
    position: 'relative',
    backgroundColor: '#000',
  },
  previewImage: {
    width: '100%',
    height: '100%',
  },
  scannerLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 4,
    backgroundColor: '#10b981',
    shadowColor: '#10b981',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
    elevation: 5,
  },
  loaderOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.5)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  loaderText: {
    color: '#fff',
    marginTop: Spacing.three,
    fontWeight: 'bold',
    fontSize: 16,
  },
  previewActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: Spacing.three,
    gap: Spacing.three,
  },
  previewActionBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.two,
    paddingVertical: Spacing.two + 2,
    borderRadius: Spacing.three,
    flex: 1,
  },
  previewActionBtnText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 14,
  },
  resultsWrapper: {
    marginTop: Spacing.five,
  },
  sectionTitle: {
    fontSize: 18,
    marginBottom: Spacing.three,
  },
  healthyCard: {
    flexDirection: 'row',
    padding: Spacing.four,
    borderRadius: Spacing.three,
    alignItems: 'center',
    gap: Spacing.three,
  },
  healthyInfo: {
    flex: 1,
  },
  healthyTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#10b981',
    marginBottom: Spacing.one,
  },
  detectionCard: {
    borderRadius: Spacing.three,
    padding: Spacing.four,
    marginBottom: Spacing.three,
    borderLeftWidth: 5,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: Spacing.two,
  },
  cardTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.two,
  },
  detectionClassName: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  confidenceBadge: {
    paddingHorizontal: Spacing.two,
    paddingVertical: Spacing.half,
    borderRadius: Spacing.two,
  },
  confidenceText: {
    fontSize: 12,
    fontWeight: 'bold',
  },
  progressBarBg: {
    height: 4,
    backgroundColor: 'rgba(0,0,0,0.05)',
    borderRadius: 2,
    marginBottom: Spacing.four,
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 2,
  },
  recommendationCard: {
    backgroundColor: 'rgba(0,0,0,0.02)',
    padding: Spacing.three,
    borderRadius: Spacing.two,
    gap: Spacing.one,
  },
  recommendationLabel: {
    fontSize: 13,
  },
  recommendationDetails: {
    fontSize: 12,
    marginTop: Spacing.one,
  },
  locationMetadata: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.one,
    marginTop: Spacing.three,
  },
  locationText: {
    fontSize: 10,
  },
});
