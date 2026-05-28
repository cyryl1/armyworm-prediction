import React, { useCallback, useEffect, useState } from 'react';
import { View, Text, FlatList, ActivityIndicator, StyleSheet, Platform } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { ThemedView } from '@/components/themed-view';
import { ThemedText } from '@/components/themed-text';
import { useTheme } from '@/hooks/use-theme';
import { SafeAreaView } from 'react-native-safe-area-context';
import { loadApiSettings } from '@/lib/api-settings';

export default function HistoryScreen() {
  const theme = useTheme();
  const [apiUrl, setApiUrl] = useState<string>(Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000');
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [records, setRecords] = useState<any[]>([]);
  const [settingsReady, setSettingsReady] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const settings = await loadApiSettings();
        setApiUrl(settings.apiUrl);
        setApiKey(settings.apiKey);
      } catch (e) {
        // ignore
      } finally {
        setSettingsReady(true);
      }
    })();
  }, []);

  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      const headers: any = { accept: 'application/json' };
      if (apiKey) headers['x-api-key'] = apiKey;
      const resp = await fetch(`${apiUrl}/history`, { headers });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      setRecords(data.records || []);
    } catch (e) {
      console.warn('Failed to load history', e);
    } finally {
      setLoading(false);
    }
  }, [apiUrl, apiKey]);

  useFocusEffect(
    useCallback(() => {
      if (settingsReady) {
        fetchHistory();
      }
    }, [fetchHistory, settingsReady])
  );

  return (
    <ThemedView style={styles.container}>
        <SafeAreaView>
            <ThemedText type="subtitle">Detection History</ThemedText>
            {loading ? (
            <ActivityIndicator />
            ) : (
                <FlatList
                    data={records}
                    keyExtractor={(item) => item.id || JSON.stringify(item)}
                    renderItem={({ item }) => (
                        <View style={styles.card}>
                        <Text style={styles.ts}>{item.detection_timestamp}</Text>
                        {item.detections ? (
                            item.detections.map((d: any, i: number) => (
                            <View key={i} style={styles.row}>
                                <Text style={styles.class}>{d.class_name} ({Math.round(d.confidence * 100)}%)</Text>
                                <Text style={styles.reco}>{d.recommendation}</Text>
                            </View>
                            ))
                        ) : (
                            <Text style={styles.class}>No detection details</Text>
                        )}
                        </View>
                    )}
                />
            )}
        </SafeAreaView>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16 },
  card: { padding: 12, marginBottom: 12, borderRadius: 8, backgroundColor: '#fff' },
  ts: { fontSize: 12, color: '#666' },
  row: { flexDirection: 'row', justifyContent: 'space-between' },
  class: { fontWeight: '600' },
  reco: { color: '#333' },
});