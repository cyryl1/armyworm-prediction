import React, { useState, useRef } from 'react';
import { View, Button, Image as RNImage, StyleSheet, Alert, Platform } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { ThemedView } from '@/components/themed-view';
import { ThemedText } from '@/components/themed-text';
import { SafeAreaView } from 'react-native-safe-area-context';
import { apiUrlToWsUrl, loadApiSettings, withWsApiKey } from '@/lib/api-settings';

export default function StreamScreen() {
  const [wsUrl, setWsUrl] = useState<string>(apiUrlToWsUrl(Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000'));
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const [annotated, setAnnotated] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [settingsReady, setSettingsReady] = useState(false);

  React.useEffect(() => {
    (async () => {
      try {
        const settings = await loadApiSettings();
        setApiKey(settings.apiKey);
        setWsUrl(apiUrlToWsUrl(settings.apiUrl));
      } catch (e) {
        // ignore
      } finally {
        setSettingsReady(true);
      }
    })();
  }, []);

  const connect = () => {
    if (!settingsReady) {
      Alert.alert('Please wait', 'Backend settings are still loading.');
      return;
    }
    try {
      const url = withWsApiKey(wsUrl, apiKey);
      wsRef.current = new WebSocket(url);
      wsRef.current.onopen = () => setConnected(true);
      wsRef.current.onmessage = (ev) => {
        try {
          const data = JSON.parse(ev.data);
          if (data.annotated_frame) {
            setAnnotated(data.annotated_frame);
          }
        } catch (e) {
          console.warn('WS message parse error', e);
        }
      };
      wsRef.current.onclose = () => setConnected(false);
      wsRef.current.onerror = (e) => {
        console.warn('WS error', e);
        setConnected(false);
      };
    } catch (e) {
      Alert.alert('WebSocket', 'Failed to connect: ' + String(e));
    }
  };

  const disconnect = () => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
  };

  const pickAndSend = async () => {
    try {
      const res = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ['images'],
        quality: 0.8,
        base64: true,
      } as any);
      if (res.canceled || !res.assets || res.assets.length === 0) return;
      const asset = res.assets[0];
      const b64 = asset.base64;
      if (!b64) return;
      const payload = JSON.stringify({ frame: b64 });
      wsRef.current?.send(payload);
    } catch (e) {
      Alert.alert('Error', 'Failed to pick/send image');
    }
  };

  return (
    <ThemedView style={styles.container}>
        <SafeAreaView>
            <ThemedText type="subtitle">Live Stream (manual frames)</ThemedText>
            <View style={styles.actions}>
                <Button title={connected ? 'Disconnect' : 'Connect'} onPress={connected ? disconnect : connect} />
                <Button title="Pick & Send Frame" onPress={pickAndSend} disabled={!connected} />
            </View>
            {annotated ? (
                <RNImage source={{ uri: `data:image/jpeg;base64,${annotated}` }} style={styles.image} />
            ) : (
                <ThemedText type="small">No annotated frame received yet.</ThemedText>
            )}
        </SafeAreaView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16 },
  actions: { flexDirection: 'row', justifyContent: 'space-between', marginVertical: 12 },
  image: { width: '100%', height: 300, resizeMode: 'contain' },
});
