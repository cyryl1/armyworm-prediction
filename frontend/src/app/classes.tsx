import React, { useEffect, useState } from 'react';
import { View, FlatList, ActivityIndicator, StyleSheet, Platform } from 'react-native';
import { ThemedView } from '@/components/themed-view';
import { ThemedText } from '@/components/themed-text';
import { SafeAreaView } from 'react-native-safe-area-context';
import { loadApiSettings } from '@/lib/api-settings';

export default function ClassesScreen() {
  const [classes, setClasses] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);
  const [apiUrl, setApiUrl] = useState<string>(Platform.OS === 'android' ? 'http://10.0.2.2:8000' : 'http://localhost:8000');
  const [settingsReady, setSettingsReady] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const settings = await loadApiSettings();
        setApiUrl(settings.apiUrl);
      } catch (e) {
        // keep default apiUrl
      } finally {
        setSettingsReady(true);
      }
    })();
  }, []);

  useEffect(() => {
    if (settingsReady) {
      fetchClasses(apiUrl);
    }
  }, [settingsReady, apiUrl]);

  const fetchClasses = async (url: string) => {
    setLoading(true);
    try {
      const resp = await fetch(`${url}/classes`, { headers: { accept: 'application/json' } });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();
      setClasses(data.classes || {});
    } catch (e) {
      console.warn('Failed to load classes', e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemedView style={styles.container}>
        <SafeAreaView>
            <ThemedText type="subtitle">Supported Classes</ThemedText>
            {loading ? (
                <ActivityIndicator />
            ) : (
                <FlatList
                data={Object.entries(classes)}
                keyExtractor={(item) => item[0]}
                renderItem={({ item }) => (
                    <View style={styles.row}>
                    <ThemedText type="default">{item[0]}</ThemedText>
                    <ThemedText type="small" themeColor="textSecondary">{item[1]}</ThemedText>
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
  row: { paddingVertical: 8, borderBottomWidth: 1, borderColor: '#eee' },
});
