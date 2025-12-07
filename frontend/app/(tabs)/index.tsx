import { useState, useEffect } from 'react';
import { Image, StyleSheet, Platform, View, Text, ScrollView, TouchableOpacity, Dimensions, StatusBar } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { FadeInDown, FadeInUp, useAnimatedStyle, useSharedValue, withRepeat, withSequence, withTiming, withSpring } from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { Ionicons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');

export default function HomeScreen() {
  const [image, setImage] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  // Animation values
  const scanLineY = useSharedValue(0);

  useEffect(() => {
    if (loading) {
      scanLineY.value = withRepeat(
        withSequence(
          withTiming(0, { duration: 0 }),
          withTiming(280, { duration: 1500 })
        ),
        -1,
        false
      );
    } else {
      scanLineY.value = 0;
    }
  }, [loading]);

  const scanLineStyle = useAnimatedStyle(() => ({
    transform: [{ translateY: scanLineY.value }],
    opacity: loading ? 1 : 0,
  }));

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setResult(null);
    }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      alert('Camera permission is required!');
      return;
    }

    let result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setResult(null);
    }
  };

  const analyzeImage = async () => {
    if (!image) return;

    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append('image', {
      uri: image,
      name: 'food.jpg',
      type: 'image/jpeg',
    } as any);

    try {
      // Use LAN IP address
      const apiUrl = 'http://10.7.1.84:5001/predict';

      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const data = await response.json();

      if (response.ok) {
        // Add a small delay for the animation effect if response is too fast
        setTimeout(() => {
          setResult(data);
          setLoading(false);
        }, 800);
      } else {
        alert('Error: ' + (data.error || 'Something went wrong'));
        setLoading(false);
      }
    } catch (error) {
      console.error(error);
      alert('Network Error: Check backend connection.');
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar barStyle="light-content" />
      <LinearGradient
        colors={['#1a2a6c', '#b21f1f', '#fdbb2d']}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.background}
      />

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <Animated.View entering={FadeInUp.delay(100).springify()} style={styles.header}>
          <View style={styles.headerIconContainer}>
            <Ionicons name="nutrition" size={32} color="#fff" />
          </View>
          <View>
            <Text style={styles.headerTitle}>Carbs AI</Text>
            <Text style={styles.headerSubtitle}>Smart Nutrition Tracker</Text>
          </View>
        </Animated.View>

        <Animated.View entering={FadeInUp.delay(300).springify()} style={styles.card}>
          <View style={styles.imageWrapper}>
            {image ? (
              <Image source={{ uri: image }} style={styles.image} />
            ) : (
              <LinearGradient colors={['#f3f4f6', '#e5e7eb']} style={styles.placeholder}>
                <Ionicons name="camera-outline" size={64} color="#9ca3af" />
                <Text style={styles.placeholderText}>Tap to Scan Food</Text>
              </LinearGradient>
            )}

            {loading && (
              <Animated.View style={[styles.scanLine, scanLineStyle]}>
                <LinearGradient
                  colors={['transparent', 'rgba(52, 211, 153, 0.8)', 'transparent']}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 0 }}
                  style={{ flex: 1 }}
                />
              </Animated.View>
            )}
          </View>

          <View style={styles.actionRow}>
            <TouchableOpacity onPress={pickImage} style={styles.iconButton}>
              <Ionicons name="images-outline" size={24} color="#374151" />
            </TouchableOpacity>

            {!loading && !result && image && (
              <TouchableOpacity onPress={analyzeImage} style={styles.analyzeBtn}>
                <LinearGradient
                  colors={['#3B82F6', '#2563EB']}
                  style={styles.gradientBtn}
                >
                  <Text style={styles.analyzeText}>Analyze Now</Text>
                  <Ionicons name="sparkles" size={18} color="#fff" />
                </LinearGradient>
              </TouchableOpacity>
            )}

            {!image && (
              <TouchableOpacity onPress={takePhoto} style={styles.mainActionBtn}>
                <LinearGradient
                  colors={['#10B981', '#059669']}
                  style={styles.gradientBtn}
                >
                  <Text style={styles.analyzeText}>Take Photo</Text>
                  <Ionicons name="camera" size={20} color="#fff" />
                </LinearGradient>
              </TouchableOpacity>
            )}

            <TouchableOpacity onPress={takePhoto} style={styles.iconButton}>
              <Ionicons name="camera-reverse-outline" size={24} color="#374151" />
            </TouchableOpacity>
          </View>
        </Animated.View>

        {result && !loading && (
          <Animated.View entering={FadeInDown.springify()} style={styles.resultContainer}>
            <BlurView intensity={80} style={styles.blurContainer}>
              <View style={styles.resultHeader}>
                <Text style={styles.dishName}>{result.dish}</Text>
                <View style={styles.confidenceBadge}>
                  <Text style={styles.confidenceText}>{result.confidence.toFixed(1)}% Match</Text>
                </View>
              </View>

              <View style={styles.statsGrid}>
                <View style={[styles.statCard, { backgroundColor: '#EFF6FF' }]}>
                  <Text style={[styles.statLabel, { color: '#1E40AF' }]}>Net Carbs</Text>
                  <Text style={[styles.statValue, { color: '#1E3A8A' }]}>{result.net_carbs.toFixed(1)}<Text style={styles.unit}>g</Text></Text>
                </View>
                <View style={[styles.statCard, { backgroundColor: '#ECFDF5' }]}>
                  <Text style={[styles.statLabel, { color: '#065F46' }]}>GI Index</Text>
                  <Text style={[styles.statValue, { color: '#064E3B' }]}>{result.glycemic_index || 'N/A'}</Text>
                </View>
              </View>

              {result.health_analysis && (
                <View style={styles.healthSection}>
                  <Text style={styles.sectionTitle}>Medical Analysis</Text>

                  <View style={[styles.infoRow, { backgroundColor: result.health_analysis.gi_analysis.risk_color === 'green' ? '#ecfdf5' : result.health_analysis.gi_analysis.risk_color === 'red' ? '#fef2f2' : '#fffbeb' }]}>
                    <Ionicons name="fitness" size={24} color={result.health_analysis.gi_analysis.risk_color === 'green' ? '#059669' : result.health_analysis.gi_analysis.risk_color === 'red' ? '#dc2626' : '#d97706'} />
                    <View style={{ marginLeft: 10, flex: 1 }}>
                      <Text style={styles.infoLabel}>{result.health_analysis.gi_analysis.category}</Text>
                      <Text style={styles.infoSub}>{result.health_analysis.gi_analysis.metabolic_impact}</Text>
                    </View>
                  </View>

                  <View style={styles.expertCard}>
                    <Text style={styles.expertTitle}>Expert Guidance</Text>

                    <Text style={styles.conditionLabel}>FOR DIABETICS:</Text>
                    <Text style={styles.conditionText}>{result.health_analysis.gi_analysis.diabetes_guidance}</Text>

                    <Text style={styles.conditionLabel}>KIDNEY SAFETY:</Text>
                    <Text style={styles.conditionText}>{result.health_analysis.specifics.kidney_safety}</Text>

                    <Text style={styles.conditionLabel}>PORTION SIZE:</Text>
                    <Text style={styles.conditionText}>{result.health_analysis.specifics.portion_size}</Text>

                    <Text style={styles.conditionLabel}>VERDICT:</Text>
                    <Text style={[styles.conditionText, { fontWeight: 'bold', color: '#111827' }]}>{result.health_analysis.specifics.recommendation.toUpperCase()}</Text>
                  </View>
                </View>
              )}

              {result.glycemic_load && (
                <View style={styles.infoRow}>
                  <View style={[styles.indicator, { backgroundColor: result.glycemic_load > 20 ? '#EF4444' : '#10B981' }]} />
                  <View>
                    <Text style={styles.infoLabel}>Glycemic Load: {result.glycemic_load.toFixed(1)}</Text>
                    <Text style={styles.infoSub}>{result.gl_class}</Text>
                  </View>
                </View>
              )}

              {result.insulin_dose !== null && (
                <LinearGradient colors={['#FFF1F2', '#FFE4E6']} style={styles.insulinCard}>
                  <View style={styles.insulinHeader}>
                    <Ionicons name="medical" size={24} color="#BE123C" />
                    <Text style={styles.insulinTitle}>Insulin Rec.</Text>
                  </View>
                  <Text style={styles.insulinValue}>{result.insulin_dose} Units</Text>
                  <Text style={styles.insulinDisclaimer}>Based on 1:12 carb ratio. Consult a doctor.</Text>
                </LinearGradient>
              )}
            </BlurView>
          </Animated.View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#111827',
  },
  background: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    height: '100%',
  },
  scrollContent: {
    padding: 20,
    paddingTop: 60,
    paddingBottom: 40,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 30,
  },
  headerIconContainer: {
    width: 50,
    height: 50,
    borderRadius: 15,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 15,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.3)',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '800',
    color: '#fff',
    letterSpacing: 0.5,
  },
  headerSubtitle: {
    fontSize: 14,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 2,
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: 24,
    padding: 15,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 10,
    marginBottom: 20,
  },
  imageWrapper: {
    width: '100%',
    height: 300,
    borderRadius: 20,
    overflow: 'hidden',
    backgroundColor: '#f3f4f6',
    position: 'relative',
  },
  image: {
    width: '100%',
    height: '100%',
  },
  placeholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    marginTop: 10,
    color: '#9ca3af',
    fontSize: 16,
    fontWeight: '500',
  },
  scanLine: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 2,
    zIndex: 10,
  },
  actionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 15,
    paddingHorizontal: 10,
  },
  iconButton: {
    width: 45,
    height: 45,
    borderRadius: 25,
    backgroundColor: '#f3f4f6',
    justifyContent: 'center',
    alignItems: 'center',
  },
  mainActionBtn: {
    flex: 1,
    marginHorizontal: 15,
    height: 50,
  },
  analyzeBtn: {
    flex: 1,
    marginHorizontal: 15,
    height: 50,
  },
  gradientBtn: {
    flex: 1,
    borderRadius: 25,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 8,
  },
  analyzeText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  resultContainer: {
    marginTop: 10,
    borderRadius: 24,
    overflow: 'hidden',
  },
  blurContainer: {
    backgroundColor: 'rgba(255,255,255,0.95)',
    padding: 20,
    borderRadius: 24,
  },
  resultHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  dishName: {
    fontSize: 24,
    fontWeight: '800',
    color: '#111827',
    textTransform: 'capitalize',
    maxWidth: '65%',
  },
  confidenceBadge: {
    backgroundColor: '#ecfdf5',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#d1fae5',
  },
  confidenceText: {
    color: '#059669',
    fontSize: 12,
    fontWeight: '600',
  },
  statsGrid: {
    flexDirection: 'row',
    gap: 15,
    marginBottom: 20,
  },
  statCard: {
    flex: 1,
    padding: 15,
    borderRadius: 16,
    alignItems: 'center',
  },
  statLabel: {
    fontSize: 13,
    fontWeight: '600',
    marginBottom: 4,
  },
  statValue: {
    fontSize: 28,
    fontWeight: '800',
  },
  unit: {
    fontSize: 14,
    fontWeight: '500',
  },
  healthSection: {
    marginTop: 10,
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#111827',
    marginBottom: 10,
  },
  expertCard: {
    backgroundColor: '#f9fafb',
    padding: 15,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#e5e7eb',
    marginTop: 10,
  },
  expertTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#374151',
    marginBottom: 10,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  conditionLabel: {
    fontSize: 11,
    fontWeight: '700',
    color: '#6b7280',
    marginTop: 8,
    marginBottom: 2,
  },
  conditionText: {
    fontSize: 13,
    color: '#374151',
    lineHeight: 18,
  },
  infoRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#f9fafb',
    padding: 15,
    borderRadius: 16,
    marginBottom: 15,
  },
  indicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 10,
  },
  infoLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: '#374151',
  },
  infoSub: {
    fontSize: 13,
    color: '#6b7280',
  },
  insulinCard: {
    padding: 15,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#fecdd3',
    alignItems: 'center',
  },
  insulinHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 5,
  },
  insulinTitle: {
    color: '#9f1239',
    fontWeight: '700',
    fontSize: 14,
  },
  insulinValue: {
    color: '#be123c',
    fontSize: 32,
    fontWeight: '800',
    marginVertical: 5,
  },
  insulinDisclaimer: {
    color: '#9f1239',
    fontSize: 11,
    opacity: 0.8,
  },
});
